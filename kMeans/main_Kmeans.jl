# Load data
using JLD
using PyPlot
pygui(true)
X = load("clusterData.jld","X")
include("clustering2Dplot.jl")
include("kMeans.jl")

# K-means clustering
let
kValues = zeros(10)
Errors = zeros(10)
minError = 100000
minModel = nothing
for k in 1:10
    for i in 1:50
        model = kMeans(X,k,doPlot=false)
        y = model.predict(X)
        error = kMeansError(X,y,model.W)
        if error < minError
            minModel = model
            minError = error
        end
        kValues[k] = k
        Errors[k] = minError
    end
end

@show kValues
@show Errors
plot(kValues, Errors)
end
#include("clustering2Dplot.jl")
#clustering2Dplot(X,y,model.W)
