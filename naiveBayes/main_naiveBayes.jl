using Printf
using Statistics
using JLD
using SparseArrays

data = load("newsgroups.jld")
X = data["X"]
y = data["y"]

Xtest = data["Xtest"]
ytest = data["ytest"]
wordlist = data["wordlist"]
groupnames = data["groupnames"]

# Compute test error with naive Bayes
include("naiveBayes.jl")
model = naiveBayes(X,y)
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test error with naive Bayes: %.3f\n",testError)
