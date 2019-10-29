include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin

	# intialize our result array
	predictionsArr = []
	# calculate the distance matrix for all points in Xhat with all points in X
  distMatrix = distancesSquared(Xhat,X)

	# Each row in the distance matrix corrosponds to the distance between
	# one Xhat point with n X points.
  for i = 1:t
		# For each row, we find the the top k indices with minimum values.
    labelIndices = sortperm(distMatrix[i,:])[1:k]
		# initialize new array for storing labels
		labels = []
		# loop through labelIndices, and append corrosponding labels to
		# the labels array
		for j = 1:length(labelIndices)
			index = labelIndices[j]
			append!(labels, y[index])
		end
		# find the mode of the labels
		prediction = mode(labels)
		# append this prediction to the predictionsArr
		append!(predictionsArr, prediction)
	end

  return predictionsArr
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end

function cknn(X,y,k)
	# Implementation of condensed k-nearest neighbour classifier
	(n,d) = size(X)
	Xcond = X[1,:]'
	ycond = [y[1]]
	for i in 2:n
    		yhat = knn_predict(X[i,:]',Xcond,ycond,k)
    		if y[i] != yhat[1]
			Xcond = vcat(Xcond,X[i,:]')
			push!(ycond,y[i])
    		end
	end

	predict(Xhat) = knn_predict(Xhat,Xcond,ycond,k)
	return GenericModel(predict)
end
