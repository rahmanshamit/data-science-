using LinearAlgebra
include("misc.jl")


#Polynomial Kernel
function leastSquaresKernelBasis(X,Xtest,y,p,lambda)
	K = polyKernelKtest(X,X,p)

	u = (K + lambda*I)\y

	predict(xhat) = polyKernelKtest(X,xhat,p)*u

	return LinearModel(predict,u)
end

function polyKernelKtest(X,xhat,p)
	n = length(X)
	m = length(xhat)
	Kcalc = zeros(m,n)
	for i in 1:m
		for j in 1:n
			Kcalc[i,j]=(1 + ( xhat[i]' * X[j] ) )^p
		end
	end
	return Kcalc
end
