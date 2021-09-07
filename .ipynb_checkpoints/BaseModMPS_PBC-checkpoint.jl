using LinearAlgebra, StatsBase, ProgressMeter
using TensorOperations

struct Index
    dim::Array{Int64}
    name::Array{String}
end

mutable struct Tensor
    data::Array{Float64}
    index::Index
end

struct MPS
    site::Array{Tensor}
    siteIndex::Array{Integer}
    MPS(A,range) = new(A,collect(range))
end

function generate_MPS(func::Function; D=5, spin_deg=2, mps_size=12)::MPS
    
    makeTensor(theIndex) = Tensor( func(Float64,tuple(theIndex.dim[:]...)),theIndex )
    
#     mps = MPS(
#         vcat(
#             vcat(
#                    [makeTensor(Index([1,D,spin_deg],["ileft","i1","s1"]))],
#                    [makeTensor(Index([D,D,spin_deg],["i$(i-1)","i$(i)","s$i"])) for i in 2:mps_size-1]
#             ),
#             [makeTensor(Index([D,1,spin_deg],["i$(mps_size-1)","iright","s$(mps_size)"]))]
#         ),
#         1:mps_size
#     )
    mps = MPS(
        vcat(
            vcat(
                   [makeTensor(Index([D,D,spin_deg],["ileft","i1","s1"]))],
                   [makeTensor(Index([D,D,spin_deg],["i$(i-1)","i$(i)","s$i"])) for i in 2:mps_size-1]
            ),
            [makeTensor(Index([D,D,spin_deg],["i$(mps_size-1)","iright","s$(mps_size)"]))]
        ),
        1:mps_size
    )
    return mps
end

function generate_Heisenberg_Hamiltonian()
    σx = [0  1;1  0]
    iσy = [0 1;-1  0]
    σz = [1  0;0  -1]
    self_kron(A::Matrix) = tensorproduct(A,(11,19),A,(21,29),(11,21,19,29))
    H = 0.25*(self_kron(σx)-self_kron(iσy)+self_kron(σz))
    return H # J=1
end

function generate_Heisenberg_Hamiltonian(Jx,Jy,Jz)
    σx = [0  1;1  0]
    iσy = [0 1;-1  0]
    σz = [1  0;0  -1]
    self_kron(A::Matrix) = tensorproduct(A,(11,19),A,(21,29),(11,21,19,29))
    # bug??
    H = 0.25*(Jx*self_kron(σx)-Jy*self_kron(iσy)+Jz*self_kron(σz))
    return H
end

function generate_eye()
    eye = [1 0;0 1]
    self_kron(A::Matrix) = tensorproduct(A,(11,19),A,(21,29),(11,21,19,29))
    eye2 = 1.0*self_kron(eye)
    return eye2
end

function sparse_indices(H)::Dict
    C1 = CartesianIndices(H[:,:,1,1])
    C2 = CartesianIndices(H[1,1,:,:])
    D = Dict{Tuple{Int8,Int8},Array{Tuple{Int8,Int8,Float64},1}}()
    for i in C1
        tmp = Tuple{Int8,Int8,Float64}[]
        # adding diagonal index(j=i) first
        push!(tmp,(i.I...,H[i,i]))
        for j in C2
            if H[i,j] != 0.0 && i != j
                push!(tmp,(j.I...,H[i,j]))
            end
        end
        push!(D, i.I => tmp)
    end
    return D
end

# right regularization:
# BBB  B
# BBB RA
# BBB' A
# BBRA A
# BB' AA
function right_regularize!(mps::MPS)
    arrS = mps.site
    len = length(arrS)
    for i in len:-1:2
        A = arrS[i].data
        B = arrS[i-1].data
        l, r, s = arrS[i].index.dim
        F = lq( reshape(A,l,r*s) )
        if l <= r*s
            A[:] = reshape(Matrix(F.Q),l,r,s)
            # the "l r s" below are just simbols in the macro, they are not what we defined above.
            @tensor C[l,r',s] := B[l,r,s]*F.L[r,r']
        else
            A[:] = reshape(Matrix(I,l,r*s)*F.Q,l,r,s)
            L = 0.0*Matrix(I,l,l)
            L[:,1:r*s] = F.L
            @tensor C[l,r',s] := B[l,r,s]*L[r,r']
        end
        B[:] = C
    end
    arrS[1].data[:] /= norm(arrS[1].data,2)
#     normalize!(arrS[1].data)
end

function exact_energy!(mps::MPS)::Real
    right_regularize!(mps)
    S = mps.site
    Et = 0.0
    A = S[1].data
    B = S[2].data
    # all real numbers in A, B, H.
    @tensoropt Ei[:] := A[l,Lu,s11]*B[Lu,r,s21]*H[s11,s21,s12,s22]*A[l,Ld,s12]*B[Ld,r,s22]
    Et += scalar(Ei)
    for i in 2:length(S)-1
        P = S[i-1].data
        A = S[i].data
        B = S[i+1].data
        l, r, s = S[i-1].index.dim
        M = tensorcopy(P,(1,2,3),(1,3,2))
        F = qr( reshape(M,l*s,r) )
        if l*s < r
            P2 = reshape( F.Q*Matrix(I,l*s,r), l,s,r )
            R = 0.0*Matrix(I,r,r)
            R[1:l*s,:] = F.R
            @tensor C[l',r,s] := R[l',l]*A[l,r,s]
        else
            P2 = reshape( Matrix(F.Q),l,s,r )
            R = F.R
            @tensor C[l',r,s] := R[l',l]*A[l,r,s]
        end
        tensorcopy!(P2,(1,3,2),P,(1,2,3))
        A[:] = C
        @tensoropt Ei[:] := A[l,Lu,s11]*B[Lu,r,s21]*H[s11,s21,s12,s22]*A[l,Ld,s12]*B[Ld,r,s22]
        Et += scalar(Ei)
    end
    return Et
end

function exact_energy(mps1::MPS)::Real
    mps = deepcopy(mps1)
    right_regularize!(mps)
    S = mps.site
    Et = 0.0
    A = S[1].data
    B = S[2].data
    # all real numbers in A, B, H.
    @tensoropt Ei[:] := A[l,Lu,s11]*B[Lu,r,s21]*H[s11,s21,s12,s22]*A[l,Ld,s12]*B[Ld,r,s22]
    Et += scalar(Ei)
    for i in 2:length(S)-1
        P = S[i-1].data
        A = S[i].data
        B = S[i+1].data
        l, r, s = S[i-1].index.dim
        M = tensorcopy(P,(1,2,3),(1,3,2))
        F = qr( reshape(M,l*s,r) )
        if l*s < r
            P2 = reshape( F.Q*Matrix(I,l*s,r), l,s,r )
            R = 0.0*Matrix(I,r,r)
            R[1:l*s,:] = F.R
            @tensor C[l',r,s] := R[l',l]*A[l,r,s]
        else
            P2 = reshape( Matrix(F.Q),l,s,r )
            R = F.R
            @tensor C[l',r,s] := R[l',l]*A[l,r,s]
        end
        tensorcopy!(P2,(1,3,2),P,(1,2,3))
        A[:] = C
        @tensoropt Ei[:] := A[l,Lu,s11]*B[Lu,r,s21]*H[s11,s21,s12,s22]*A[l,Ld,s12]*B[Ld,r,s22]
        Et += scalar(Ei)
    end
    return Et
end

function exact_Sz(mps1::MPS)::Real
    mps = deepcopy(mps1)
    right_regularize!(mps)
    S = mps.site
    len = length(S)
    Szt = 0.0
    A = S[1].data
#     B = S[2].data
    Ŝz = Matrix(0.5*Diagonal( (spin_deg-1):-2:-(spin_deg-1) ) )
    # all real numbers in A, B, H.
    @tensoropt Sz[:] := A[l,r,s1]*Ŝz[s1,s2]*A[l,r,s2]
    Szt += scalar(Sz)
    for i in 2:len
        P = S[i-1].data
        A = S[i].data
#         B = S[i+1].data
        l, r, s = S[i-1].index.dim
        M = tensorcopy(P,(1,2,3),(1,3,2))
        F = qr( reshape(M,l*s,r) )
        if l*s < r
            P2 = reshape( F.Q*Matrix(I,l*s,r), l,s,r )
            R = 0.0*Matrix(I,r,r)
            R[1:l*s,:] = F.R
            @tensor C[l',r,s] := R[l',l]*A[l,r,s]
        else
            P2 = reshape( Matrix(F.Q),l,s,r )
            R = F.R
            @tensor C[l',r,s] := R[l',l]*A[l,r,s]
        end
        tensorcopy!(P2,(1,3,2),P,(1,2,3))
        A[:] = C
        @tensoropt Sz[:] := A[l,r,s1]*Ŝz[s1,s2]*A[l,r,s2]
        Szt += scalar(Sz)
    end
    return Szt/len
end

# PBC_only, sqrt(<S|S>)
function exact_norm(mps::MPS)::Real
    S = mps.site
    len = length(S)
    A = S[1].data
    @tensor T[l1,l2,r1,r2] := A[l1,r1,s]*A[l2,r2,s]
    for i in 2:len
        A = S[i].data
        @tensoropt R[l1,l2,r1,r2] := T[l1,l2,z1,z2]*A[z1,r1,s]*A[z2,r2,s]
        T = R
    end
    @tensor trace[:] := T[z1,z2,z1,z2]
    return sqrt(scalar(trace))
end

function print_mat(A::AbstractArray)
    show(stdout,"text/plain",A)
    println(stdout,"\n")
end

# H = generate_Heisenberg_Hamiltonian()
# spH = sparse_indices(H)

mutable struct Sample
    state::Vector{Int8}
    inner::Float64   #W(S)
    energy::Float64  #E(S)
end

mutable struct Node
    spin::Int8
    value::Float64
    parent::Int32
end

function generate_BTree(s::Int)::Vector{Node}
    l = 2^(s+1)-2
    W = Vector{Node}(undef,l)
    for i in 1:2:l-1
        W[i] = Node(1,0.0,(i-1)/2)
    end
    for i in 2:2:l
        W[i] = Node(2,0.0,i/2-1)
    end
    return W
end
    
# configure next sample state in Markov chain 
# function next(tmp::Vector{Int8})
#     l = length(tmp)
#     new = deepcopy(tmp)
#     for i in 1:2:l-1
#         if tmp[i] != tmp[i+1]
#             new[i] = tmp[i+1]
#             new[i+1] = tmp[i]
#         end
#     end
#     selected = StatsBase.sample(1:l,Int(ceil(l/10)),replace=false,ordered=true)
#     new[selected] = 3 .- new[selected]
#     return new
# end
function next(tmp::Vector{Int8})
    l = length(tmp)
    new = deepcopy(tmp)
    for i in 1:2:l-1
        if tmp[i] != tmp[i+1]
            new[i] = tmp[i+1]
            new[i+1] = tmp[i]
        end
    end
    selected = StatsBase.sample(1:l,1,replace=false,ordered=true)
    new[selected] = 3 .- new[selected]
    return new
end

function state_inner(tmp::Sample,mps::MPS)
    l = length(tmp.state)
    if length(mps.site) != l
        throw(DomainError( (l,length(mps.site)), "length dismatch" ))
    end
    matchain = Array{Matrix{Float64},1}(undef,l)
    for i in 1:l
        spin = tmp.state[i]
        matchain[i] = mps.site[i].data[:,:,spin]
    end
    inner = matchain[1]
    for i in 2:l
        inner = inner*matchain[i]
    end
    return tr(inner)
end

function state_inner(state::Vector{Int8},mps::MPS)
    l = length(state)
    if length(mps.site) != l
        throw(DomainError( (l,length(mps.site)), "length dismatch" ))
    end
    matchain = Array{Matrix{Float64},1}(undef,l)
    for i in 1:l
        spin = state[i]
        matchain[i] = mps.site[i].data[:,:,spin]
    end
    inner = matchain[1]
    for i in 2:l
        inner = inner*matchain[i]
    end
    return tr(inner)
end

function calc_Es(sample::Sample, mps::MPS, spH::Dict)::Float64
    state = sample.state
    S = mps.site
    l = length(S)
    Es_diag = 0.0
    Es_offdiag = 0.0
    W = sample.inner
    for i in 1:l-1
        spin = (state[i], state[i+1])
        for (s1,s2,e) in spH[spin]
            if (s1, s2) == spin
                Es_diag += e
            else
                dual_state = deepcopy(state)
                dual_state[i:i+1] = [s1,s2]
                W′ = state_inner(dual_state,mps)
                Es_offdiag += W′*e
            end
        end
    end
    # adding the following code changes OBC to PBC
    if true
        spin = (state[l], state[1])
        for (s1,s2,e) in spH[spin]
            if (s1, s2) == spin
                Es_diag += e
            else
                dual_state = deepcopy(state)
                dual_state[l] = s1
                dual_state[1] = s2
                W′ = state_inner(dual_state,mps)
                Es_offdiag += W′*e
            end
        end
    end
    # end code
    
    return Es_diag + Es_offdiag/W
end

# not a gradient actually, it just contracts MPS's matrix except the given one(with index n).
function single_site_grad(state, mps::MPS, H, n)
    S = mps.site
    l = length(state)
    if length(S) != l
        throw(DomainError( (l,length(mps.site)), "length dismatch" ) )
    end
    matchain = Array{Matrix{Float64},1}(undef,l)
    for i in 1:l
        spin = state[i]
        matchain[i] = S[i].data[:,:,spin]
    end
    lmat = 1.0
    rmat = 1.0
    for i in 1:n-1
        lmat = lmat*matchain[i]
    end
    for i in n+1:l
        rmat = rmat*matchain[i]
    end
    G = zeros(Float64,size(S[n].data))
    G[:,:,state[n]] = lmat' * rmat'
    return G
end

function tr2vec(Tree::Vector{Node},node_number)
    i = node_number
    j = mps_size
    chain = Vector{Int8}(undef, mps_size)
    while j != 0 
        chain[end+1-j] = Tree[i].spin
        i = Tree[i].parent
        j -= 1
    end
    return chain
end

# from 2^k-1 to 2(2^k-1) are samples' <S'|H|S >. S: temporary(tmp) state.
function generate_samples!(Tree,tmp,H,mps)
    K = mps_size
    for k in 2:K
        for i in 2^k-1:2*(2^k-1)
            j = Tree[i].parent
            s11 = Tree[j].spin
            s12 = Tree[i].spin
            s21 = tmp.state[k-1]
            s22 = tmp.state[k]
            Tree[i].value = Tree[j].value +   H[s11,s12,s21,s22]
        end
    end
    samples = Vector{Sample}(undef,2^K)
    D = view(Tree,2^K-1:2*(2^K-1))
    for k in 1:2^K
        state = tr2vec(Tree,k+2^K-2)
        samples[k] = Sample(state,state_inner(state,mps),D[k].value)
        
    end
    return samples
end

function gradSampleGenerate(mps::MPS, H, chainlen)
    S = mps.site
    Ns = length(S)
    chain = rand(Int8.(1:spin_deg),Ns)
    tmp = Sample(chain,1e-5,0.0)
    new = tmp
    samples = Vector{Sample}(undef,chainlen)
    
    for i in 1:Int(ceil(chainlen/2.5))
        new = Sample(next(tmp.state), 0.0, 0.0)
        new.inner = state_inner(new, mps)
        if rand() < (new.inner/tmp.inner)^2
            tmp = new
        end
        # heating, do nothing here
    end
    
    samples = Vector{Sample}(undef,chainlen)
    Es_sum = 0.0   # ∑ E(S)
    grad = Vector{Array{Float64,3}}(undef,mps_size)
    for i in 1:mps_size
        # will be used to store ∑2/W(S)|S>
        grad[i] = zero(S[i].data)
    end
    delta = zero.(grad)
    
    tmp.energy = calc_Es(tmp, mps, spH)
    for i in 1:chainlen
        new = Sample(next(tmp.state), 0.0, 0.0)
        new.inner = state_inner(new, mps)
        if rand() < (new.inner/tmp.inner)^2
            tmp = new
            tmp.energy = calc_Es(tmp, mps, spH)
        end
        
        # tmp.energy: E(S)
        Es_sum += tmp.energy
        samples[i] = tmp
        for i in 1:mps_size
            B = 2/tmp.inner*single_site_grad(tmp.state, mps, H, i)
            delta[i] += B
            grad[i] += B*tmp.energy
        end
    end
    grad ./= chainlen
    delta ./= chainlen
    Es_sum /= chainlen
    grad = grad .- Es_sum.*delta
    return grad, Es_sum, samples
end

function exact_grad(mps::MPS, H)
    S = mps.site
    K = length(S)
    D = view(Tree,2^K-1:2*(2^K-1))
    # list all possible states in samples[], we only use samples[i].state
    samples = Vector{Sample}(undef,2^K)
    for k in 1:2^K
        state = tr2vec(Tree,k+2^K-2)
        samples[k] = Sample(state,0.0,0.0)
    end
    for tmp in samples
        # W(S)
        tmp.inner = state_inner(tmp, mps)
        # E(S)
        tmp.energy = calc_Es(tmp, mps, spH)
    end
    grad = gradFromSet(mps, H, samples)
    return grad
end

function gradFromSet(mps::MPS, mps_old::MPS, H, set)
    S_new = mps.site
    S_old = mps_old.site
    Es_sum = 0.0   # ∑ E(S)
    psum = 0.0 # Σ ρ'/ρ
    grad = Vector{Array{Float64,3}}(undef,mps_size)
    for i in 1:mps_size
        # ∑ 2/W(S)|S>
        grad[i] = zero(S_new[i].data)
    end
    delta = zero.(grad)
    for sp in set
        rate = (state_inner(sp,mps)/sp.inner)^2
        psum += rate
        
        energy1 = calc_Es(sp, mps, spH)
        Es_sum += energy1*rate
        
        for i in 1:mps_size
            B = 2/sp.inner*single_site_grad(sp.state, mps, H, i)
            delta[i] += B*rate
            grad[i] += B*sp.energy*rate
        end
    end
    grad ./= psum
    delta ./= psum
    Es_sum /= psum
    grad = grad .- Es_sum.*delta
    return grad
end

function gradFromSet(mps::MPS, H, set)
    S = mps.site
    Es_sum = 0.0   # ∑ E(S)
    grad = Vector{Array{Float64,3}}(undef,mps_size)
    for i in 1:mps_size
        # ∑ 2/W(S)|S>
        grad[i] = zero(S[i].data)
    end
    delta = zero.(grad)
    for sp in set
        Es_sum += sp.energy
        
        for i in 1:mps_size
            B = 2/sp.inner*single_site_grad(sp.state, mps, H, i)
            delta[i] += B
            grad[i] += B*sp.energy
        end
    end
    chainlen = length(set)
    grad ./= chainlen
    delta ./= chainlen
    Es_sum /= chainlen
    grad = grad .- Es_sum.*delta
    return grad
end

# PBC
function mkSGD(mps1::MPS, H; step=20, chainlen=200, η=0.005, progress=false)
    mps = deepcopy(mps1)
    val = Vector{Float64}(undef,step)
    if progress
        progress1 = Progress(step)
    end
    energy1 = 0.0
    
    for j in 1:step
        if j%100 == 0 
            right_regularize!(mps)
        end
        mps.site[1].data /= exact_norm(mps)
        
        grad, energy1, _ = gradSampleGenerate(mps, H, chainlen)
        g = grad
        for i in 1:mps_size
#             mps.site[i].data[:] = mps.site[i].data - η*g[i]
            mps.site[i].data[:] = mps.site[i].data - η*(1/(i/5000+1))*g[i]
        end
#         val[j+1] = exact_energy(mps)
        val[j] = energy1
        progress && next!(progress1)
    end
    return val, mps
end

# PBC
function mkAdam(mps1::MPS, H; step=20, chainlen=200, η=0.005, β1=0.9, β2=0.999, ϵ=1e-8, progress=false)
    mps = deepcopy(mps1)
    val = Vector{Float64}(undef,step)
#     val[1] = exact_energy!(mps)
    if progress
        progress1 = Progress(step)
    end
    energy1 = 0.0
    grad, _, _ = gradSampleGenerate(mps, H, 2)
    g = zero.(grad)
    v = zero.(g)
    for j in 1:step
#         if j%100 == 0 
#             right_regularize!(mps)
#         end
#         mps.site[1].data /= exact_norm(mps)
        
        grad, energy1, _ = gradSampleGenerate(mps, H, chainlen)
        for i in 1:mps_size
            g[i] = β1*g[i]+(1-β1)*grad[i]
            v[i] = β2*v[i]+(1-β2)*grad[i].^2
            ηt = η*sqrt(1-β2^j)/(1-β1^j)
#             mps.site[i].data[:] = mps.site[i].data - η*g[i]
            mps.site[i].data[:] = mps.site[i].data - ηt*(g[i]./(sqrt.(v[i]).+ϵ*sqrt(1-β2^j)))
        end
#         val[j+1] = exact_energy(mps)
        val[j] = energy1
        progress && next!(progress1)
    end
    return val, mps
end

function mkReweightSVRG(mps1::MPS, H; outer=20,inner=10,long=200,short=40,η=0.005, progress=false)
    mps = deepcopy(mps1)
    val = Vector{Float64}(undef,inner*outer+1)
    # cannot obtain exact energy in PBC
    val[1] = exact_energy!(mps)
    if progress
        progress1 = Progress(outer)
    end
    energy1 = 0.0
    for i in 1:outer
        if i%100 == 0 
            right_regularize!(mps)
        end
        mps.site[1].data /= exact_norm(mps)
        
        mps_old = deepcopy(mps)
        grad, energy1, set = gradSampleGenerate(mps, H, long)
        for j in 1:inner
            
#             _, subset = gradSampleGenerate(mps,H,short)
            partlink = StatsBase.sample(1:long,short,replace=false,ordered=true)
            subset = set[partlink]
            
            g = grad-gradFromSet(mps_old,H,subset)+gradFromSet(mps_old,mps,H,subset)
            for i in 1:mps_size
                mps.site[i].data[:] = mps.site[i].data - η*g[i]
            end
#             val[(i-1)*inner+j+1] = exact_energy(mps)
            val[(i-1)*inner+j+1] = energy1
        end
        progress && next!(progress1)
    end
    return val, mps
end

function tgReweightSVRG(mps1::MPS, H; outer=2,inner=10,long=200,short=40,η=0.005, progress=false)
    mps = deepcopy(mps1)
    val = Vector{Float64}(undef,inner*outer+1)
    # cannot obtain exact energy in PBC
    val[1] = exact_energy!(mps)
    if progress
        progress1 = Progress(outer)
    end
    energy1 = 0.0
    for i in 1:outer
        mps_old = deepcopy(mps)
        _, energy1, set = gradSampleGenerate(mps, H, long)
        grad = exact_grad(mps, H)
        for j in 1:inner
#             _, subset = gradSampleGenerate(mps,H,short)
            partlink = StatsBase.sample(1:long,short,replace=false,ordered=true)
            subset = set[partlink]
            g = grad-gradFromSet(mps_old,H,subset)+gradFromSet(mps_old,mps,H,subset)
            for i in 1:mps_size
                mps.site[i].data[:] = mps.site[i].data - η*g[i]
            end
#             val[(i-1)*inner+j+1] = exact_energy(mps)
            val[(i-1)*inner+j+1] = energy1
        end
        progress && next!(progress1)
    end
    return val, mps
end
