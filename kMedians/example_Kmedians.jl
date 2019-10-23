# Load data
using JLD
using PyPlot
pygui(true)
X = load("clusterData2.jld","X")
include("kMedians.jl")
include("clustering2Dplot.jl")

# K-means clustering
let
#kValues = zeros(10)
#Errors = zeros(10)
minError = 100000
minModel = nothing
k =4

for i in 1:50
    model = kMedians(X,k,doPlot=false)
    y = model.predict(X)
    error = kMediansError(X,y,model.W)
    if error < minError
        minModel = model
        minError = error
    end
end

#plot(kValues, Errors)
#end
include("clustering2Dplot.jl")
clustering2Dplot(X,minModel.predict(X),minModel.W)
end
