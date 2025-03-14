function initialise(M::AbstractMatrix, A::SparseMatrixCSC, b::AbstractVector)
  mrow, mcol = size(M)
  n = mrow
  if mrow != mcol
    throw(error("M is not sconstNumuare"))
  end
  if !issymmetric(M)
    throw(error("M is not symmetric"))
  end

  constNum = length(b)
  arow, acol = size(A)
  if constNum != acol
    throw(error("A and b have different sizes"))
  end

  T = promote_type(Float32, eltype(M), eltype(A), eltype(b))
  r = min(n, constNum)

  x = zeros(T, n)
  δx = zeros(T, n)
  λ = zeros(T, constNum)
  δλ = zeros(T, r)
  d = zeros(T, n)

  L = zeros(T, n, n)
  J₀ = zeros(T, n, n)
  J = zeros(T, n, n)
  R = zeros(T, div(r * (r + 1), 2))

  constNorms = zeros(T, constNum)
  constValues = zeros(T, constNum)
  activeConstIdx = zeros(Int, constNum)
  activeλ = zeros(T, r + 1)

  iter = zeros(Int, 2)

  cache = (;
    A, b,
    x, δx, λ, δλ, d,
    L, J₀, J, R,
    constNorms, constValues,
    activeConstIdx, activeλ,
    iter
  )

  updateM!(cache, M)
  updateA!(cache, A)

  cache
end

function updateM!(cache, M::AbstractMatrix)
  L, J = cache.L, cache.J₀

  # Factorise M into Cholesky form
  copy!(L, M)
  L, info = choleskyFactorise!(L)
  if !iszero(info)
    throw(error("M is not positive definite"))
  end

  # Inverse L
  copy!(J, L)
  choleskyInverse!(J)

  # Reset upper part of J
  n = size(J, 1)
  for col in 1:n
    for row in col+1:n
      J[row, col] = 0
    end
  end

  cache
end

function updateMDiag!(cache, M::AbstractMatrix)
  L, J = cache.L, cache.J₀

  # Factorise M into Cholesky form
  copy!(L, M)
  for i in axes(M, 1)
    Mii = M[i, i]
    (Mii <= 0) && throw(error("M is not positive definite"))
    L[i, i] = sqrt(Mii)
  end

  # Inverse L
  copy!(J, L)
  for i in axes(M, 1)
    J[i, i] = inv(L[i, i])
  end

  cache
end

function updateA!(cache, A::SparseMatrixCSC)
  constNorms = cache.constNorms
  vals = nonzeros(A)

  # Compute the constraint norms
  for j in axes(A, 2)
    Σ² = zero(eltype(constNorms))
    for k in nzrange(A, j)
      Σ² += vals[k]^2
    end
    isnan(Σ²) && throw(DomainError(Σ²))
    constNorms[j] = sqrt(Σ²)
  end

  cache
end

function solve!(v::AbstractVector, cache)
  cache, info = qpgen!(v, cache)

  if !iszero(info)
    throw(error("Inconsistent constraints, no x"))
  end

  v, cache
end

function qpgen!(v::AbstractVector, cache)
  A, b = cache.A, cache.b
  x, δx = cache.x, cache.δx
  λ, δλ = cache.λ, cache.δλ
  d = cache.d
  L, J₀, J, R = cache.L, cache.J₀, cache.J, cache.R
  constNorms, constValues = cache.constNorms, cache.constValues
  activeConstIdx, activeλ = cache.activeConstIdx, cache.activeλ
  iter = cache.iter

  n = length(v)
  constNum = length(b)
  r = min(n, constNum)
  T = eltype(x)
  ε = 2 * eps(T)

  Arows = rowvals(A)
  Avals = nonzeros(A)

  # Reset J
  copy!(J, J₀)

  # Reset active constraints
  fill!(activeConstIdx, 0)
  activeConstNum = 0

  ##########
  # Step 0 #
  ##########
  # Slightly messy, store v in d
  copy!(d, v)

  # Compute unconstrained solution
  choleskySolve!(v, L)

  # Messy store v in x
  copy!(x, v)

  # Compute unconstrained energy
  energy = -dot(d, x) / 2

  # Reset d
  fill!(d, 0)

  ierr = 0
  iter[1] = 0
  iter[2] = 0

  ########################
  # Evaluate constraints #
  ########################
  @label LEvaluateConstraints

  iter[1] += 1

  # Evaluate constraints
  for c in 1:constNum
    Σ = -b[c]
    for k in nzrange(A, c)
      Σ += Avals[k] * x[Arows[k]]
    end
    constValues[c] = (abs(Σ) < ε) ? 0 : Σ
  end

  # Force active constraints to zero
  for k in 1:activeConstNum
    constValues[activeConstIdx[k]] = 0
  end

  # Choose next constraint
  nextConstIdx = 0
  nextConstScore = zero(T)
  for c in 1:constNum
    if constValues[c] < nextConstScore * constNorms[c]
      nextConstIdx = c
      nextConstScore = constValues[c] / constNorms[c]
      if iszero(constValues[c])
        nextConstScore = zero(T)
      end
    end
  end

  # Terminate if all constraints are satisfied
  if iszero(nextConstIdx)
    fill!(λ, 0)
    for c in 1:activeConstNum
      λ[activeConstIdx[c]] = activeλ[c]
    end

    copy!(v, x)
    return cache, ierr
  end

  ###########################
  # Fix selected constraint #
  ###########################
  @label LFixConstraint

  # Compute d = J' A[nextConstIdx, :]
  for i in 1:n
    Σ = zero(T)
    for k in nzrange(A, nextConstIdx)
      Σ += J[Arows[k], i] * Avals[k]
    end
    isnan(Σ) && throw(DomainError(Σ))
    d[i] = Σ
  end

  # Compute δx = J[:, inactive] * d[inactive]
  fill!(δx, 0)
  for j in activeConstNum+1:n
    for i in 1:n
      δx[i] += J[i, j] * d[j]
    end
  end

  # Compute R^{-1} d_1 and find maximum step size t₁ that does not violate the
  # constraints and the corresponding constraint number t₁Idx
  t₁ = T(Inf)
  t₁Idx = 1
  t₁Found = false
  for i in activeConstNum:-1:1
    Σ = d[i]
    l = div(i * (i + 3), 2)
    l₁ = l - i
    for j in i+1:activeConstNum
      Σ -= R[l] * δλ[j]
      l += j
    end
    if !iszero(Σ)
      Σ /= R[l₁]
    end
    isnan(Σ) && throw(DomainError(Σ))
    δλ[i] = Σ
    (Σ <= 0) && continue

    t₁Found = true
    t = activeλ[i] / δλ[i]
    if t < t₁
      t₁ = t
      t₁Idx = i
    end
  end

  if sum(abs2, δx) <= ε
    # No step in primal space such that the new constraint becomes feasible
    # Take step in dual space and drop a constraint
    if !t₁Found
      # No step in dual space possible either, problem is not solvable
      ierr = 1
      return cache, ierr
    else
      # Take partial step in dual space and drop constraint t₁Idx
      # Then continue with step 2(a) (marked by label 55)
      for i = 1:activeConstNum
        activeλ[i] -= t₁ * δλ[i]
      end
      activeλ[activeConstNum+1] += t₁
      @goto LDropConstraint
    end
  else
    # Compute full step length t₂, minimum step in primal space such that
    # the constraint becomes feasible
    # Keep σ = δx^T n to update energy
    σ = zero(T)
    for k in nzrange(A, nextConstIdx)
      σ += δx[Arows[k]] * Avals[k]
    end

    t₂ = -constValues[nextConstIdx] / σ
    t₂min = true
    if t₁Found && (t₁ < t₂)
      t₂ = t₁
      t₂min = false
    end

    # Take step in primal and dual space
    for i in 1:n
      x[i] += t₂ * δx[i]
    end
    energy += t₂ * σ * (t₂ / 2 + activeλ[activeConstNum+1])
    for i in 1:activeConstNum
      activeλ[i] -= t₂ * δλ[i]
    end
    activeλ[activeConstNum+1] += t₂

    # * If it was a full step, check wheter further constraints are violated
    # * Otherwise drop the current constraint and iterate once more
    if t₂min
      # Took a full step

      # Add constraint `nextConstIdx` to the list of active constraint
      activeConstNum += 1
      activeConstIdx[activeConstNum] = nextConstIdx

      # Update R
      # Need to put the first `activeConstNum` - 1 components of `d`
      # into column `activeConstNum` of R
      l = div((activeConstNum - 1) * activeConstNum, 2) + 1
      for i = 1:activeConstNum-1
        R[l] = d[i]
        l += 1
      end

      # * If `activeConstNum` = `n`, then only need to add the last element to the new
      # row of R.
      # * Otherwise use Givens transformations to turn the vector d[activeConstNum:n]
      # into a multiple of the first unit vector. That multiple goes into the
      # last element of the new row of R and J is accordingly updated by the
      # Givens transformations.
      if activeConstNum == n
        R[l] = d[n]
      else
        for i in n:-1:activeConstNum+1
          # Find the Givens rotation which reduces the element `l₁` of d to zero
          # If it is already zero we don't have to do anything, except decreasing l₁
          if iszero(d[i])
            continue
          end
          sθ, cθ = minmax(abs(d[i]), abs(d[i-1]))
          isnan(sθ) && throw(DomainError(sθ))
          isnan(cθ) && throw(DomainError(cθ))
          radius = copysign(sqrt(sθ^2 + cθ^2), d[i-1])
          isnan(radius) && throw(DomainError(radius))
          sinθ = d[i] / radius
          cosθ = d[i-1] / radius

          # The Givens rotation is done with the matrix (cosθ sinθ, sinθ -cosθ)
          # * If cosθ = 1, then d[i] << d[l₁ - 1] so no need to do anything
          # * If cosθ = 0, then switch columns i and i - 1 of J.
          # Since we only switch columns in J, we have to be careful how we
          # update d depending on the sign of sinθ.
          # * Otherwise we have to apply the Givens rotation to these columns.
          # The i-1 element of d has to be updated to `radius`.
          if isone(cosθ)
            continue
          end
          if iszero(cosθ)
            d[i-1] = radius * sign(sinθ)
            for j in 1:n
              J[j, i], J[j, i-1] = J[j, i-1], J[j, i]
            end
          else
            d[i-1] = radius
            ν = sinθ / (1 + cosθ)
            for j in 1:n
              temp = cosθ * J[j, i-1] + sinθ * J[j, i]
              J[j, i] = ν * (J[j, i-1] + temp) - J[j, i]
              J[j, i-1] = temp
            end
          end
        end

        # l is still pointing to element (activeConstNum, activeConstNum) in R
        # Store d[activeConstNum] in R[activeConstNum, activeConstNum]
        R[l] = d[activeConstNum]
      end
    else
      # Partial step in dual space
      # Drop constraint t₁Idx, then continue with step 2(a)
      σ = -b[nextConstIdx]
      for k in nzrange(A, nextConstIdx)
        σ += x[Arows[k]] * Avals[k]
      end
      constValues[nextConstIdx] = σ
      @goto LDropConstraint
    end
  end
  @goto LEvaluateConstraints

  ############################
  # Drop selected constraint #
  ############################
  @label LDropConstraint

  # * If t₁Idx = activeConstNum, it is only necessary to update the lagrange vector
  # and activeConstNum
  # * Otherwise, need to update R and J
  if t₁Idx == activeConstNum
    @goto LRemoveConstraintWithoutJR
  end

  ##################
  # Update J and R #
  ##################
  @label LUpdateJR
  # Will also come back here after updating one row of R (column of J)

  # Find the Givens rotation that eliminates the pivot t₁Idx from R
  # If this pivot is already zero, only need to update
  # u, iact, and shifting column (t₁Idx+1) of R to column (t₁Idx)
  # l  will point to element (1, t₁Idx+1) of R
  # l₁ will point to element (t₁Idx+1, t₁Idx+1) of R
  l = div(t₁Idx * (t₁Idx + 1), 2) + 1
  l₁ = l + t₁Idx
  if iszero(R[l₁])
    @goto LRemoveConstraintWithJR
  end
  sθ, cθ = minmax(abs(R[l₁-1]), abs(R[l₁]))
  isnan(sθ) && throw(DomainError(sθ))
  isnan(cθ) && throw(DomainError(cθ))
  radius = copysign(sqrt(sθ^2 + cθ^2), R[l₁-1])
  isnan(radius) && throw(DomainError(radius))
  sinθ = R[l₁] / radius
  cosθ = R[l₁-1] / radius

  # The Givens rotatin is done with the matrix (gc gs, gs -gc).
  # * If gc = 1, then R[t₁Idx+1, t₁Idx+1] << R[t₁Idx,t₁Idx+1], no need to do anything
  # * If gc = 0, then switch rows `t₁Idx` and `t₁Idx+1`
  # of R and column `t₁Idx` and `t₁Idx+1` of J. Since we switch rows in
  # R and columns in J, we can ignore the sign of gs.
  # * Otherwise we have to apply the Givens rotation to these rows/columns.
  if isone(gc)
    @goto LRemoveConstraintWithJR
  end
  if iszero(gc)
    for i = t₁Idx+1:activeConstNum
      R[l₁], R[l₁-1] = R[l₁-1], R[l₁]
      l₁ = l₁ + i
    end
    for i = 1:n
      J[i, t₁Idx], J[i, t₁Idx+1] = J[i, t₁Idx+1], J[i, t₁Idx]
    end
  else
    ν = gs / (1 + gc)
    if iszero(gs)
      ν = zero(T)
    end
    for i = t₁Idx+1:activeConstNum
      temp = gc * R[l₁-1] + gs * R[l₁]
      R[l₁] = ν * (R[l₁-1] + temp) - R[l₁]
      R[l₁-1] = temp
      l₁ = l₁ + i
    end
    for i = 1:n
      temp = gc * J[i, t₁Idx] + gs * J[i, t₁Idx+1]
      J[i, t₁Idx+1] = ν * (J[i, t₁Idx] + temp) - J[i, t₁Idx+1]
      J[i, t₁Idx] = temp
    end
  end

  #####################
  # Remove constraint #
  #####################
  # Shift column `t₁Idx+1` to `t₁Idx` in R (the first t₁Idx elements)
  # The position of element (1, t₁Idx+1) of R was calculated above and stored in l
  @label LRemoveConstraintWithJR
  l₁ = l - t₁Idx
  for i = 1:t₁Idx
    R[l₁] = R[l]
    l = l + 1
    l₁ = l₁ + 1
  end

  # Update lagrange vector and active constant index
  # Continue with updating the matrices J and R
  activeλ[t₁Idx] = activeλ[t₁Idx+1]
  activeConstIdx[t₁Idx] = activeConstIdx[t₁Idx+1]
  t₁Idx += 1
  if t₁Idx < activeConstNum
    @goto LUpdateJR
  end

  @label LRemoveConstraintWithoutJR
  activeλ[activeConstNum] = activeλ[activeConstNum+1]
  activeλ[activeConstNum+1] = zero(T)
  activeConstIdx[activeConstNum] = 0
  activeConstNum -= 1
  iter[2] += 1
  @goto LFixConstraint

  copy!(v, x)
  return cache, ierr
end
