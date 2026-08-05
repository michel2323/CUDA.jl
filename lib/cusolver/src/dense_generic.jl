# Xpotrf
function Xpotrf!(uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXpotrf_bufferSize(dh, params, uplo, n,
                                    T, A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXpotrf(dh, params, uplo, n, T, A, lda, T,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, flag
end

# Xpotrs
function Xpotrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    p, nrhs = size(B)
    (p ≠ n) && throw(DimensionMismatch("first dimension of B, $p, must match second dimension of A, $n"))
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    cusolverDnXpotrs(dh, params, uplo, n, nrhs, T, A, lda, T, B, ldb, dh.info)

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

# Xgetrf
function Xgetrf!(A::StridedCuMatrix{T}, ipiv::CuVector{Int64}) where {T <: BlasFloat}
    m, n = size(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgetrf_bufferSize(dh, params, m, n, T,
                                    A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgetrf(dh, params, m, n, T, A, lda, ipiv,
                         T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, ipiv, flag
end

function Xgetrf!(A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m,n = size(A)
    ipiv = CuVector{Int64}(undef, min(m, n))
    Xgetrf!(A, ipiv)
end

# Xgetrs
function Xgetrs!(trans::Char, A::StridedCuMatrix{T}, ipiv::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    chktrans(trans)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    cusolverDnXgetrs(dh, params, trans, n, nrhs, T, A, lda, ipiv, T, B, ldb, dh.info)

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

# Xgeqrf
function Xgeqrf!(A::StridedCuMatrix{T}, tau::CuVector{T}) where {T <: BlasFloat}
    m, n = size(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeqrf_bufferSize(dh, params, m, n, T, A,
                                    lda, T, tau, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeqrf(dh, params, m, n, T, A,
                         lda, T, tau, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, tau
end

function Xgeqrf!(A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    tau = CuVector{T}(undef, min(m,n))
    Xgeqrf!(A, tau)
end

# Xsytrs
function sytrs!(uplo::Char, A::StridedCuMatrix{T}, p::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dh, uplo, n, nrhs, T, A,
                                    lda, p, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu,
                    bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dh, uplo, n, nrhs, T, A, lda, p,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

function sytrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dh, uplo, n, nrhs, T, A,
                                    lda, CU_NULL, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu,
                    bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dh, uplo, n, nrhs, T, A, lda, CU_NULL,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

# Xtrtri
function trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    chkdiag(diag)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXtrtri_bufferSize(dh, uplo, diag, n, T, A, lda, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXtrtri(dh, uplo, diag, n, T, A, lda,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu),
                         dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A
end

# Xlarft!
function larft!(direct::Char, storev::Char, v::StridedCuMatrix{T}, tau::StridedCuVector{T}, t::StridedCuMatrix{T}) where {T <: BlasFloat}
    cuSOLVER.version() < v"11.6.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    n, k = size(v)
    ktau = length(tau)
    mt, nt = size(t)
    (storev != 'C') && throw(ArgumentError("Only storev = 'C' is supported."))
    (n < k) && throw(ArgumentError("The number of elementary reflectors ($k) must be lower or equal to the order of block reflector H ($n)."))
    (ktau != k) && throw(ArgumentError("The length of tau ($ktau) is not equal to the number of elementary reflectors ($k)."))
    (mt != k || nt != k) && throw(ArgumentError("The size of the triangular factor of the block reflector is ($mt, $nt) and must be ($k, $k)."))
    ldv = max(1, stride(v, 2))
    ldt = max(1, stride(t, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXlarft_bufferSize(dh, params, direct, storev, n, k, T,
                                    v, ldv, T, tau, T, t, ldt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXlarft(dh, params, direct, storev, n, k, T, v, ldv, T, tau, T, t,
                         ldt, T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu))
    end

    t
end

# larfb!
#
# cuSOLVER has no `Xlarfb`, so the block reflector is applied with cuBLAS. All of
# the operations below are GEMMs and TRMMs, which take 64-bit dimensions, so this
# works for matrices of any size.
"""
    larfb!(side, trans, direct, storev, V, t, C)
    larfb!(side, trans, direct, storev, V, t, C, work)

Apply the block reflector `H = I - V * t * Vᴴ` to `C`, overwriting it with `H * C`
(`side = 'L'`, `trans = 'N'`), `Hᴴ * C` (`side = 'L'`, `trans = 'T'` or `'C'`),
`C * H` (`side = 'R'`, `trans = 'N'`) or `C * Hᴴ` (`side = 'R'`, `trans = 'T'` or
`'C'`).

`V` holds the elementary reflectors in unit lower trapezoidal form and `t` is the
triangular factor computed by [`larft!`](@ref). Only `direct = 'F'` and
`storev = 'C'` are supported. `work` is a scratch matrix of size `(k, n)` for
`side = 'L'` and `(m, k)` for `side = 'R'`, where `(m, n) = size(C)` and
`k = size(V, 2)`; it is allocated on the fly if not provided.
"""
function larfb!(side::Char, trans::Char, direct::Char, storev::Char,
                V::StridedCuMatrix{T}, t::StridedCuMatrix{T},
                C::StridedCuMatrix{T}, work::StridedCuMatrix{T}) where {T <: BlasFloat}
    # Support trans = 'C' for real matrices
    trans = T <: Real && trans == 'C' ? 'T' : trans
    (T <: Complex) && (trans == 'T') && throw(ArgumentError("trans = 'T' is not supported with complex matrices."))
    chkside(side)
    chktrans(trans)
    (direct != 'F') && throw(ArgumentError("Only direct = 'F' is supported."))
    (storev != 'C') && throw(ArgumentError("Only storev = 'C' is supported."))

    m, n = size(C)
    nv, k = size(V)
    mt, nt = size(t)
    (mt != k || nt != k) && throw(DimensionMismatch("the triangular factor of the block reflector is ($mt, $nt) and must be ($k, $k)."))
    if nv != (side == 'L' ? m : n)
        throw(DimensionMismatch("the reflectors have $nv rows, which must match the $(side == 'L' ? "first" : "second") dimension of C, $(side == 'L' ? m : n)."))
    end
    mw, nw = size(work)
    if (side == 'L' && (mw < k || nw < n)) || (side == 'R' && (mw < m || nw < k))
        throw(DimensionMismatch("the workspace is ($mw, $nw) and must be at least $(side == 'L' ? "($k, $n)" : "($m, $k)")."))
    end
    k == 0 && return C

    # H = I - V t Vᴴ, so Hᴴ = I - V tᴴ Vᴴ
    transt = trans == 'N' ? 'N' : (T <: Real ? 'T' : 'C')
    transv = T <: Real ? 'T' : 'C'

    if side == 'L'
        W = view(work, 1:k, 1:n)
        cuBLAS.gemm!(transv, 'N', one(T), V, C, zero(T), W)      # W  = Vᴴ C
        cuBLAS.trmm!('L', 'U', transt, 'N', one(T), t, W, W)     # W  = op(t) W
        cuBLAS.gemm!('N', 'N', -one(T), V, W, one(T), C)         # C -= V W
    else
        W = view(work, 1:m, 1:k)
        cuBLAS.gemm!('N', 'N', one(T), C, V, zero(T), W)         # W  = C V
        cuBLAS.trmm!('R', 'U', transt, 'N', one(T), t, W, W)     # W  = W op(t)
        cuBLAS.gemm!('N', transv, -one(T), W, V, one(T), C)      # C -= W Vᴴ
    end

    C
end

function larfb!(side::Char, trans::Char, direct::Char, storev::Char,
                V::StridedCuMatrix{T}, t::StridedCuMatrix{T},
                C::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkside(side)
    m, n = size(C)
    k = size(V, 2)
    work = side == 'L' ? similar(C, k, n) : similar(C, m, k)
    try
        larfb!(side, trans, direct, storev, V, t, C, work)
    finally
        unsafe_free!(work)
    end
end

_unit_lower(v, i, j) = i > j ? v : (i == j ? one(v) : zero(v))

"""
    unit_lower_triangular!(V)

Overwrite `V` in place with its unit lower trapezoidal part: the strict upper
triangle is set to zero and the diagonal to one.

[`geqrf!`](@ref) leaves R in the upper triangle of the matrix that stores the
elementary reflectors, whereas [`larft!`](@ref) and [`larfb!`](@ref) expect the
reflectors in unit lower trapezoidal form.
"""
function unit_lower_triangular!(V::StridedCuMatrix)
    m, n = size(V)
    V .= _unit_lower.(V, 1:m, (1:n)')
    V
end

# Default block size of the blocked Householder implementations below, chosen as
# a compromise: a larger value amortises the per-block `larft!` call and kernel
# launches over more columns, but `larft!` itself costs O(m * k * blocksize), so
# past some point it dominates. Measured on an RTX 4080 for m = k = 8192, the
# optimum is around 512 in Float32 and around 128 in Float64; 128 is within a few
# percent of cuSOLVER's legacy `ormqr` in both. The workspace grows linearly with
# this value, so even large block sizes stay negligible.
const WY_BLOCKSIZE = 128

# `larft!`, and therefore everything built on top of it, needs CUDA 12.4
has_blocked_householder() = cuSOLVER.version() >= v"11.6.0"

"""
    Xormqr!(side, trans, A, tau, C; blocksize=$(WY_BLOCKSIZE))

Multiply `C` in place by the orthogonal (unitary) matrix `Q` of the QR
factorization stored in `A` and `tau`, as returned by [`Xgeqrf!`](@ref) or
[`geqrf!`](@ref): `Q * C`, `Qᴴ * C`, `C * Q` or `C * Qᴴ` depending on `side` and
`trans`.

This is a blocked Householder (compact WY) implementation on top of
[`larft!`](@ref) and cuBLAS. It exists because cuSOLVER only provides `ormqr` in
its legacy 32-bit interface, which wants a workspace growing like `O(m * k)` and
reports its size as a 32-bit element count; it therefore refuses any problem
needing more than `typemax(Cint)` workspace elements — a square QR of about
32500x32500 and up — however much device memory is available. This routine needs
only `O((m + n) * blocksize)` scratch space and places no 32-bit limit on the
problem size.

See also [`ormqr!`](@ref), which uses cuSOLVER's legacy routine and falls back to
this one for problems it cannot handle.
"""
function Xormqr!(side::Char, trans::Char, A::StridedCuMatrix{T}, tau::StridedCuVector{T},
                 C::StridedCuVecOrMat{T}; blocksize::Integer=WY_BLOCKSIZE) where {T <: BlasFloat}
    # Support trans = 'C' for real matrices
    trans = T <: Real && trans == 'C' ? 'T' : trans
    (T <: Complex) && (trans == 'T') && throw(ArgumentError("trans = 'T' is not supported with complex matrices."))
    chkside(side)
    chktrans(trans)
    (blocksize < 1) && throw(ArgumentError("blocksize must be positive."))

    m, n = ndims(C) == 2 ? size(C) : (length(C), 1)
    mA = size(A, 1)
    k = length(tau)
    if side == 'L' && m != mA
        throw(DimensionMismatch("for a left-sided multiplication, the first dimension of C, $m, must equal the first dimension of A, $mA"))
    end
    if side == 'R' && n != mA
        throw(DimensionMismatch("for a right-sided multiplication, the second dimension of C, $n, must equal the first dimension of A, $mA"))
    end
    if side == 'L' && k > m
        throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= m = $m"))
    end
    if side == 'R' && k > n
        throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= n = $n"))
    end
    k == 0 && return C

    Cm = ndims(C) == 2 ? C : reshape(C, m, 1)
    nb = min(blocksize, k)
    V = similar(A, mA, nb)                                    # panel of reflectors
    t = similar(A, nb, nb)                                    # its triangular factor
    work = side == 'L' ? similar(A, nb, n) : similar(A, m, nb)

    # Q = H(1) H(2) ... H(k), so Qᴴ * C and C * Q consume the blocks front to
    # back while Q * C and C * Qᴴ consume them back to front.
    forward = (side == 'L') ⊻ (trans == 'N')
    offsets = forward ? (0:nb:k-1) : reverse(0:nb:k-1)

    try
        for j in offsets
            jb = min(nb, k - j)
            # H(j+1) ... H(j+jb) only acts on rows j+1:mA
            Vj = view(V, 1:(mA-j), 1:jb)
            copyto!(Vj, view(A, (j+1):mA, (j+1):(j+jb)))
            unit_lower_triangular!(view(Vj, 1:jb, 1:jb))
            tj = view(t, 1:jb, 1:jb)
            larft!('F', 'C', Vj, view(tau, (j+1):(j+jb)), tj)

            Cj = side == 'L' ? view(Cm, (j+1):mA, :) : view(Cm, :, (j+1):mA)
            larfb!(side, trans, 'F', 'C', Vj, tj, Cj, work)
        end
    finally
        unsafe_free!(V)
        unsafe_free!(t)
        unsafe_free!(work)
    end

    C
end

"""
    Xorgqr!(A, tau; blocksize=$(WY_BLOCKSIZE))

Overwrite `A` with the first `min(size(A)...)` columns of the orthogonal
(unitary) matrix `Q` of the QR factorization stored in `A` and `tau`, as returned
by [`Xgeqrf!`](@ref) or [`geqrf!`](@ref).

Like [`Xormqr!`](@ref) this is a blocked Householder (compact WY) implementation
that avoids cuSOLVER's legacy 32-bit `orgqr`, and it does so without allocating a
second copy of `Q`.
"""
function Xorgqr!(A::StridedCuMatrix{T}, tau::StridedCuVector{T};
                 blocksize::Integer=WY_BLOCKSIZE) where {T <: BlasFloat}
    (blocksize < 1) && throw(ArgumentError("blocksize must be positive."))
    m = size(A, 1)
    n = min(m, size(A, 2))
    k = length(tau)
    (k > n) && throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= n = $n"))

    # columns k+1:n of Q are the corresponding columns of the identity matrix,
    # which the block reflectors below are then applied to
    if k < n
        view(A, :, (k+1):n) .= T.((1:m) .== ((k+1):n)')
    end

    if k > 0
        nb = min(blocksize, k)
        V = similar(A, m, nb)
        t = similar(A, nb, nb)
        Y = similar(A, nb, nb)
        work = similar(A, nb, n)

        try
            # build Q back to front, so that the columns to the right of the
            # current panel already hold H(j+jb+1) ... H(k) times the identity
            for j in reverse(0:nb:k-1)
                jb = min(nb, k - j)
                Vj = view(V, 1:(m-j), 1:jb)
                copyto!(Vj, view(A, (j+1):m, (j+1):(j+jb)))
                V1 = view(Vj, 1:jb, 1:jb)
                unit_lower_triangular!(V1)
                tj = view(t, 1:jb, 1:jb)
                larft!('F', 'C', Vj, view(tau, (j+1):(j+jb)), tj)

                if j + jb < n
                    larfb!('L', 'N', 'F', 'C', Vj, tj, view(A, (j+1):m, (j+jb+1):n), work)
                end

                # the leading jb columns of I - V t Vᴴ are [I; 0] - V * (t * V1ᴴ)
                Yj = view(Y, 1:jb, 1:jb)
                Yj .= adjoint(V1)
                cuBLAS.trmm!('L', 'U', 'N', 'N', one(T), tj, Yj, Yj)
                Aj = view(A, (j+1):m, (j+1):(j+jb))
                cuBLAS.gemm!('N', 'N', -one(T), Vj, Yj, zero(T), Aj)
                view(Aj, 1:jb, 1:jb) .+= T.((1:jb) .== (1:jb)')
                fill!(view(A, 1:j, (j+1):(j+jb)), zero(T))
            end
        finally
            unsafe_free!(V)
            unsafe_free!(t)
            unsafe_free!(Y)
            unsafe_free!(work)
        end
    end

    n < size(A, 2) ? A[:, 1:n] : A
end

# Xgesvd
function Xgesvd!(jobu::Char, jobvt::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    R = real(T)
    (m < n) && throw(ArgumentError("The number of rows of A ($m) must be greater or equal to the number of columns of A ($n)"))
    k = min(m, n)
    U = if jobu == 'A'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'S'
        CuMatrix{T}(undef, m, k)
    elseif jobu == 'N' || jobu == 'O'
        CU_NULL
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    Σ = CuVector{R}(undef, k)
    Vt = if jobvt == 'A'
        CuMatrix{T}(undef, n, n)
    elseif jobvt == 'S'
        CuMatrix{T}(undef, k, n)
    elseif jobvt == 'N' || jobvt == 'O'
        CU_NULL
    else
        throw(ArgumentError("jobvt is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldvt = Vt == CU_NULL ? 1 : max(1, stride(Vt, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvd_bufferSize(dh, params, jobu, jobvt,
                                    m, n, T, A, lda, R, Σ, T, U, ldu,
                                    T, Vt, ldvt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvd(dh, params, jobu, jobvt, m, n, T, A,
                         lda, R, Σ, T, U, ldu, T, Vt, ldvt, T, buffer_gpu,
                         sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    U, Σ, Vt
end

# Xgesvdp
function Xgesvdp!(jobz::Char, econ::Int, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    econ in (0, 1) || throw(ArgumentError("econ is incorrect. The values accepted are 0 and 1."))
    return Xgesvdp!(jobz, Bool(econ), A)
end

function Xgesvdp!(jobz::Char, econ::Bool, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    p = min(m, n)
    R = real(T)
    jobz in ('N', 'V') || throw(ArgumentError("jobz is incorrect. The values accepted are 'V' and 'N'."))

    if econ
        U = CuMatrix{T}(undef, m, p)
        V = CuMatrix{T}(undef, n, p)
    else
        U = CuMatrix{T}(undef, m, m)
        V = CuMatrix{T}(undef, n, n)
    end
    Σ = CuVector{R}(undef, p)
    lda = max(1, stride(A, 2))
    ldu = max(1, stride(U, 2))
    ldv = max(1, stride(V, 2))
    h_err_sigma = Ref{Cdouble}(0)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdp_bufferSize(dh, params, jobz, econ, m,
                                     n, T, A, lda, R, Σ, T, U, ldu, T, V,
                                     ldv, T, out_gpu, out_cpu)

        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdp(dh, params, jobz, econ, m, n, T, A, lda, R,
                          Σ, T, U, ldu, T, V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), dh.info, h_err_sigma)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    if jobz == 'N'
        unsafe_free!(U)
        unsafe_free!(V)
        return Σ, h_err_sigma[]
    elseif jobz == 'V'
        return U, Σ, V, h_err_sigma[]
    end
end

# Xgesvdr
function Xgesvdr!(jobu::Char, jobv::Char, A::StridedCuMatrix{T}, k::Integer;
                  niters::Integer=2, p::Integer=2*k) where {T <: BlasFloat}
    m, n = size(A)
    ℓ = min(m,n)
    p = min(p, ℓ-k)  # Ensure that p + k ≤ ℓ
    (1 ≤ k ≤ ℓ) || throw(ArgumentError("illegal choice of parameter k = $k, which must be between 1 and min(m,n) = $ℓ"))
    (k+p ≤ ℓ) || throw(ArgumentError("illegal choice of parameters k = $k and p = $p, which must satisfy k+p ≤ min(m,n) = $ℓ"))
    R = real(T)
    U = if jobu == 'S'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'N'
        CuMatrix{T}(undef, m, ℓ)
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'S' and 'N'."))
    end
    Σ = CuVector{R}(undef, ℓ)
    V = if jobv == 'S'
        CuMatrix{T}(undef, n, n)
    elseif jobv == 'N'
        CuMatrix{T}(undef, n, ℓ)
    else
        throw(ArgumentError("jobv is incorrect. The values accepted are 'S' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldv = V == CU_NULL ? 1 : max(1, stride(V, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdr_bufferSize(dh, params, jobu, jobv,
                                     m, n, k, p, niters, T, A, lda, R, Σ, T,
                                     U, ldu, T, V, ldv, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdr(dh, params, jobu, jobv, m, n,
                          k, p, niters, T, A, lda, R, Σ, T, U, ldu, T,
                          V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    U, Σ, V
end

# Xsyevd
function Xsyevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevd_bufferSize(dh, params, jobz, uplo, n,
                                    T, A, lda, R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevd(dh, params, jobz, uplo, n, T, A,
                         lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

# Xsyevdx
function Xsyevdx!(jobz::Char, range::Char, uplo::Char, A::StridedCuMatrix{T};
                  vl::Real=0.0, vu::Real=Inf, il::Integer=1, iu::Integer=0) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    (n ≥ 1) && (iu == 0) && (iu = n)
    (range == 'I') && !(1 ≤ il ≤ iu ≤ n) && throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
    (range == 'V') && (vl ≥ vu) && throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n)
    vl = Ref{R}(vl)
    vu = Ref{R}(vu)
    h_meig = Ref{Int64}(0)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevdx_bufferSize(dh, params, jobz, range, uplo, n,
                                     T, A, lda, vl, vu, il, iu, h_meig,
                                     R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevdx(dh, params, jobz, range, uplo, n, T, A,
                          lda, vl, vu, il, iu, h_meig, R, W, T, buffer_gpu,
                          sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    if jobz == 'N'
        return W, h_meig[]
    elseif jobz == 'V'
        return W, A, h_meig[]
    end
end

# Xgeev
function Xgeev!(jobvl::Char, jobvr::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    cuSOLVER.version() < v"11.7.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    n = checksquare(A)
    VL = if jobvl == 'V'
        CuMatrix{T}(undef, n, n)
    elseif jobvl == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvl is incorrect. The values accepted are 'V' and 'N'."))
    end
    C = T <: Real ? Complex{T} : T
    W = CuVector{C}(undef, n)
    VR = if jobvr == 'V'
        CuMatrix{T}(undef, n, n)
    elseif jobvr == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvr is incorrect. The values accepted are 'V' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldvl = VL == CU_NULL ? 1 : max(1, stride(VL, 2))
    ldvr = VR == CU_NULL ? 1 : max(1, stride(VR, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeev_bufferSize(dh, params, jobvl, jobvr, n, T, A,
                                   lda, C, W, T, VL, ldvl, T, VR, ldvr,
                                   T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeev(dh, params, jobvl, jobvr, n, T, A, lda, C,
                        W, T, VL, ldvl, T, VR, ldvr, T, buffer_gpu,
                        sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    return W, VL, VR
end

# XsyevBatched
function XsyevBatched!(jobz::Char, uplo::Char, A::StridedCuArray{T, 3}) where {T <: BlasFloat}
    minimum_version = v"11.7.1"
    cuSOLVER.version() < minimum_version && throw(ErrorException("This operation requires cuSOLVER
        $(minimum_version) or later. Current cuSOLVER version: $(cuSOLVER.version())."))
    chkuplo(uplo)
    n = checksquare(A)
    batch_size = size(A, 3)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuMatrix{R}(undef, n, batch_size)
    params = CuSolverParameters()
    dh = dense_handle()
    resize!(dh.info, batch_size)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevBatched_bufferSize(
            dh, params, jobz, uplo, n,
            T, A, lda, R, W, T, out_gpu, out_cpu, batch_size
        )
        return out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevBatched(
            dh, params, jobz, uplo, n, T, A,
            lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
            buffer_cpu, sizeof(buffer_cpu), dh.info, batch_size
        )
    end

    info = @allowscalar collect(dh.info)
    for i in 1:batch_size
        chkargsok(info[i] |> BlasInt)
    end

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

function XsyevBatched!(jobz::Char, uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    minimum_version = v"11.7.1"
    cuSOLVER.version() < minimum_version && throw(ErrorException("This operation requires cuSOLVER
        $(minimum_version) or later. Current cuSOLVER version: $(cuSOLVER.version())."))
    chkuplo(uplo)
    n, num_matrices = size(A)
    batch_size = num_matrices ÷ n
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n * batch_size)
    params = CuSolverParameters()
    dh = dense_handle()
    resize!(dh.info, batch_size)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevBatched_bufferSize(dh, params, jobz, uplo, n,
                                          T, A, lda, R, W, T, out_gpu, out_cpu, batch_size)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevBatched(dh, params, jobz, uplo, n, T, A,
                               lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
                               buffer_cpu, sizeof(buffer_cpu), dh.info, batch_size)
    end

    info = @allowscalar collect(dh.info)
    for i = 1:batch_size
        chkargsok(info[i] |> BlasInt)
    end

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

# LAPACK
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        LinearAlgebra.LAPACK.sytrs!(uplo::Char, A::StridedCuMatrix{$elty}, p::CuVector{Int64}, B::StridedCuVecOrMat{$elty}) = cuSOLVER.sytrs!(uplo, A, p, B)
        LinearAlgebra.LAPACK.trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{$elty}) = cuSOLVER.trtri!(uplo, diag, A)
    end
end
