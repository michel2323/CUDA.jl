using cuSOLVER
using LinearAlgebra

m = 15
n = 10
p = 5

@testset "trtri! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    for uplo in ('L', 'U')
        for diag in ('N', 'U')
            A = rand(elty, n, n)
            A = uplo == 'L' ? tril(A) : triu(A)
            A = diag == 'N' ? A : A - Diagonal(A) + I
            d_A = CuMatrix(A)
            d_B = copy(d_A)
            cuSOLVER.trtri!(uplo, diag, d_B)
            @test collect(d_A * d_B) ≈ I
        end
    end
end

@testset "potrf! -- potrs! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    for uplo in ('L', 'U')
        A    = rand(elty, n, n)
        A    = A*A' + I
        B    = rand(elty, n, p)
        d_A  = CuMatrix(A)
        d_B  = CuMatrix(B)

        cuSOLVER.Xpotrf!(uplo, d_A)
        cuSOLVER.Xpotrs!(uplo, d_A, d_B)
        LAPACK.potrf!(uplo, A)
        LAPACK.potrs!(uplo, A, B)
        @test B ≈ collect(d_B)
    end
end

@testset "getrf! -- getrs! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    for trans in ('N', 'T', 'C')
        A   = rand(elty, n, n)
        B   = rand(elty, n, p)
        d_A = CuMatrix(A)
        d_B = CuMatrix(B)

        d_A, d_ipiv, _ = cuSOLVER.Xgetrf!(d_A)
        cuSOLVER.Xgetrs!(trans, d_A, d_ipiv, d_B)
        A, ipiv, _ = LAPACK.getrf!(A)
        LAPACK.getrs!(trans, A, ipiv, B)
        @test B ≈ collect(d_B)
    end
end

@testset "geqrf! -- orgqr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, m, n)
    d_A = CuMatrix(A)
    d_A, tau = cuSOLVER.Xgeqrf!(d_A)
    cuSOLVER.orgqr!(d_A, tau)
    @test collect(d_A' * d_A) ≈ I
end

if cuSOLVER.version() >= v"11.6.0"
    @testset "larft! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "direct = $direct" for direct in ('F', 'B')
            direct == 'B' && continue
            A = rand(elty, m, n)
            t = rand(elty, n, n)

            dA = CuMatrix(A)
            dA, dτ = cuSOLVER.geqrf!(dA)
            hI = Matrix{elty}(I, m, m)
            dI = CuArray(hI)
            dH = cuSOLVER.ormqr!('L', 'N', dA, dτ, copy(dI))

            v = Array(dA)
            for j = 1:n
                v[j,j] = one(elty)
                for i = 1:j-1
                    v[i,j] = zero(elty)
                end
            end
            dv = CuArray(v)
            dt = CuMatrix(t)
            dt = cuSOLVER.larft!(direct, 'C', dv, dτ, dt)
            @test dI - dv * dt * dv' ≈ dH
        end
    end

    @testset "unit_lower_triangular! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        for (r, c) in ((m, n), (n, m), (n, n))
            A = rand(elty, r, c)
            reference = tril(A, -1) + Matrix{elty}(I, r, c)
            @test collect(cuSOLVER.unit_lower_triangular!(CuMatrix(A))) == reference
        end
    end

    @testset "larfb! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        trans_types = elty <: Real ? ('N', 'T', 'C') : ('N', 'C')
        dV, dτ = cuSOLVER.geqrf!(CuMatrix(rand(elty, m, p)))
        cuSOLVER.unit_lower_triangular!(dV)
        dt = cuSOLVER.larft!('F', 'C', dV, dτ, CuMatrix{elty}(undef, p, p))
        # the block reflector, formed explicitly
        H = Matrix{elty}(I, m, m) - collect(dV) * collect(dt) * collect(dV)'

        @testset "side = $side, trans = $trans" for side in ('L', 'R'), trans in trans_types
            C = side == 'L' ? rand(elty, m, n) : rand(elty, n, m)
            op = trans == 'N' ? H : H'
            dC = CuMatrix(C)
            cuSOLVER.larfb!(side, trans, 'F', 'C', dV, dt, dC)
            @test collect(dC) ≈ (side == 'L' ? op * C : C * op)
        end
    end

    @testset "Xormqr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        trans_types = elty <: Real ? ('N', 'T', 'C') : ('N', 'C')
        # tall, wide and square factorizations
        @testset "size = $((r, c))" for (r, c) in ((m, n), (n, m), (n, n))
            dA, dτ = cuSOLVER.geqrf!(CuMatrix(rand(elty, r, c)))
            A, τ = collect(dA), collect(dτ)
            # blocksize below, at and above the number of reflectors
            @testset "blocksize = $blocksize" for blocksize in (1, 3, 64)
                @testset "trans = $trans" for trans in trans_types
                    # LAPACK only accepts 'T' for real matrices, where cuSOLVER
                    # additionally supports the equivalent 'C'
                    ref_trans = elty <: Real && trans == 'C' ? 'T' : trans

                    B = rand(elty, r, p)
                    dB = CuMatrix(B)
                    cuSOLVER.Xormqr!('L', trans, dA, dτ, dB; blocksize)
                    @test collect(dB) ≈ LAPACK.ormqr!('L', ref_trans, copy(A), τ, copy(B))

                    C = rand(elty, p, r)
                    dC = CuMatrix(C)
                    cuSOLVER.Xormqr!('R', trans, dA, dτ, dC; blocksize)
                    @test collect(dC) ≈ LAPACK.ormqr!('R', ref_trans, copy(A), τ, copy(C))

                    v = rand(elty, r)
                    dv = CuVector(v)
                    cuSOLVER.Xormqr!('L', trans, dA, dτ, dv; blocksize)
                    @test collect(dv) ≈ LAPACK.ormqr!('L', ref_trans, copy(A), τ, copy(v))
                end
            end
            # the blocked implementation must agree with cuSOLVER's legacy one
            @testset "agrees with ormqr!" for trans in trans_types
                B = CuMatrix(rand(elty, r, p))
                @test collect(cuSOLVER.ormqr!('L', trans, dA, dτ, copy(B))) ≈
                      collect(cuSOLVER.Xormqr!('L', trans, dA, dτ, copy(B)))
            end
        end
    end

    @testset "Xorgqr! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "size = $((r, c))" for (r, c) in ((m, n), (n, m), (n, n))
            dA, dτ = cuSOLVER.geqrf!(CuMatrix(rand(elty, r, c)))
            A, τ = collect(dA), collect(dτ)
            @testset "blocksize = $blocksize" for blocksize in (1, 3, 64)
                dQ = cuSOLVER.Xorgqr!(copy(dA), dτ; blocksize)
                @test collect(dQ' * dQ) ≈ I
                @test collect(dQ) ≈ LAPACK.orgqr!(copy(A), τ)[:, 1:size(dQ, 2)]
            end
            # fewer reflectors than columns of Q
            k = max(1, length(τ) - 2)
            dQ = cuSOLVER.Xorgqr!(copy(dA), dτ[1:k])
            @test collect(dQ) ≈ LAPACK.orgqr!(copy(A), τ[1:k])[:, 1:size(dQ, 2)]
            # and agreement with cuSOLVER's legacy implementation
            @test collect(cuSOLVER.orgqr!(copy(dA), dτ)) ≈ collect(cuSOLVER.Xorgqr!(copy(dA), dτ))
        end
    end
end

@testset "sytrs! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "uplo = $uplo" for uplo in ('L', 'U')
        @testset "pivoting = $pivoting" for pivoting in (false, true)
            !pivoting && (cuSOLVER.version() < v"11.7.2") && continue
            # NVIDIA bug #5949478: non-pivoting sytrf performs Hermitian
            # instead of symmetric factorization for complex types on
            # CUSOLVER 12.0.9 (CUDA 13.1)
            !pivoting && elty <: Complex && v"12.0.9" <= cuSOLVER.version() && continue
            A = rand(elty, n, n)
            B = rand(elty, n, p)
            C = rand(elty, n)
            A = A + transpose(A)
            d_A = CuMatrix(A)
            d_B = CuMatrix(B)
            d_C = CuVector(C)
            if pivoting
                d_A, d_ipiv, _ = cuSOLVER.sytrf!(uplo, d_A; pivoting)
                d_ipiv = CuVector{Int64}(d_ipiv)
                cuSOLVER.sytrs!(uplo, d_A, d_ipiv, d_B)
                cuSOLVER.sytrs!(uplo, d_A, d_ipiv, d_C)
                A, ipiv, _ = LAPACK.sytrf!(uplo, A)
                LAPACK.sytrs!(uplo, A, ipiv, B)
                LAPACK.sytrs!(uplo, A, ipiv, C)
                @test B ≈ collect(d_B)
                @test C ≈ collect(d_C)
            else
                d_A, _ = cuSOLVER.sytrf!(uplo, d_A; pivoting)
                cuSOLVER.sytrs!(uplo, d_A, d_B)
                cuSOLVER.sytrs!(uplo, d_A, d_C)
                # Verify correctness directly: non-pivoting cusolver cannot be
                # compared against LAPACK (which always pivots), so instead check
                # that A * x ≈ b for the original inputs.
                @test A * collect(d_B) ≈ B
                @test A * collect(d_C) ≈ C
            end
        end
    end
end
