using LinearAlgebra
include("misc.jl")

#gaussian-RBF Kernel
function rbfKernelBasis(X,y,sigma,lambda)
	K = rbfKernel(X,X,sigma)

	u = (K + lambda*I)\y

	predict(xhat) = rbfKernel(X,xhat,sigma)*u

	return LinearModel(predict,u)
end

function rbfKernel(X,xhat,sigma)
	n = length(X)
	m = length(xhat)
	Kcalc = zeros(m,n)
	for i in 1:m
		for j in 1:n
			Kcalc[i,j] = exp(0-sum.((xhat[i].-X[j]).^2)/(2*sigma^2))
		end
	end
	return Kcalc
end







function leastSquaresBasis(x,y,p)
	Z = polyBasis(x,p)

	v = (Z'*Z)\(Z'*y)

	predict(xhat) = polyBasis(xhat,p)*v

	return LinearModel(predict,v)
end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

function weightedLeastSquares(X,y,v)
	V = diagm(v)
	w = (X'*V*X)\(X'*V*y)
	predict(Xhat) = Xhat*w
	return LinearModel(predict,w)
end

function binaryLeastSquares(X,y)
	w = (X'X)\(X'y)

	predict(Xhat) = sign.(Xhat*w)

	return LinearModel(predict,w)
end


function leastSquaresRBF(X,y,sigma)
	(n,d) = size(X)

	Z = rbf(X,X,sigma)

	v = (Z'*Z)\(Z'*y)

	predict(Xhat) = rbf(Xhat,X,sigma)*v

	return LinearModel(predict,v)
end

function rbf(Xhat,X,sigma)
	(t,d) = size(Xhat)
	n = size(X,1)
	D = distancesSquared(Xhat,X)
	return (1/sqrt(2pi*sigma^2))exp.(-D/(2sigma^2))
end
