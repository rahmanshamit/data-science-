using Printf
using Random

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)

# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
minErr = Inf
bestSigma = []
for sigma in 2.0.^(-15:15)
	mean = 0
	folds = 10
	trialErrors=zeros(folds)
	subset_size=fld(n,folds)
	lengthValid = 0

	for i in 1:folds
		perm = randperm(n)
		validStart = Int64(lengthValid+1) # Start of validation indices
		validEnd = Int64(i*(n/folds)) # End of validation incides
		validNdx = perm[validStart:validEnd] # Indices of validation examples
		lengthValid = length(validNdx) #update length
		trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples

		Xtrain = X[trainNdx,:]
		ytrain = y[trainNdx]

		Xvalid = X[validNdx,:]
		yvalid = y[validNdx]

		# Train on the training set
		model = leastSquaresRBF(Xtrain,ytrain,sigma,10^-12)

		# Compute the error on the validation set
		yhat = model.predict(Xvalid)
		validError = sum(((yhat - yvalid).^2)/(n/folds))
		trialErrors[i]=validError
	end
	@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError)

	# Keep track of the lowest validation error
	minError = trialErrors[argmin(trialErrors)]
	if minError < minErr
		global minErr = minError
		global bestSigma = sigma
	end
	@show bestSigma
end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma,10^-12)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using PyPlot
pygui(true)
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
