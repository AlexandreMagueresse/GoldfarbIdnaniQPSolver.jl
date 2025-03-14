function choleskyFactorise!(A::AbstractMatrix{T}) where {T}
  for col in axes(A, 2)
    info = col
    s = zero(T)
    for row in 1:col-1
      t = A[row, col] - dotMatColCol(A, 1, row - 1, row, col)
      t /= A[row, row]
      A[row, col] = t
      s += t^2
    end
    s = A[col, col] - s
    if s <= zero(T)
      return A, info
    end
    A[col, col] = sqrt(s)
  end
  return A, 0
end

function choleskySolve!(b::AbstractArray{T}, A::AbstractMatrix{T}) where {T}
  for row in eachindex(b)
    λ = dotMatColVec(A, b, 1, row - 1, row)
    b[row] = (b[row] - λ) / A[row, row]
  end
  for row in reverse(eachindex(b))
    b[row] = b[row] / A[row, row]
    λ = -b[row]
    axpyMatColVec!(b, λ, A, 1, row - 1, row)
  end
  b
end

function choleskyInverse!(A::AbstractMatrix)
  n = size(A, 2)
  for row in axes(A, 1)
    λ = inv(A[row, row])
    A[row, row] = λ
    rmulMatCol!(A, -λ, 1, row - 1, row)
    for col in row+1:n
      λ = A[row, col]
      A[row, col] = 0
      axpyMatColCol!(A, λ, 1, row, row, col)
    end
  end
  A
end

function dotMatColCol(A::AbstractMatrix, rowStart::Int, rowEnd::Int, col1::Int, col2::Int)
  res = zero(eltype(A))
  for row in rowStart:rowEnd
    res += A[row, col1] * A[row, col2]
  end
  return res
end

function dotMatColVec(A::AbstractMatrix, b::AbstractVector, rowStart::Int, rowEnd::Int, col::Int)
  res = zero(promote_type(eltype(A), eltype(b)))
  for row in rowStart:rowEnd
    res += A[row, col] * b[row]
  end
  return res
end

function axpyMatColCol!(A::AbstractMatrix{T}, λ::Number, rowStart::Int, rowEnd::Int, col1::Int, col2::Int) where {T}
  iszero(λ) && return A
  for row in rowStart:rowEnd
    A[row, col2] += λ * A[row, col1]
  end
  A
end

function axpyMatColVec!(b::AbstractVector, λ::Number, A::AbstractMatrix, rowStart::Int, rowEnd::Int, col::Int)
  iszero(λ) && return b
  for row in rowStart:rowEnd
    b[row] += λ * A[row, col]
  end
  return b
end

function rmulMatCol!(A::AbstractMatrix, λ::Number, rowStart::Int, rowEnd::Int, col::Int)
  for row in rowStart:rowEnd
    A[row, col] *= λ
  end
  A
end
