include("misc.jl") # Includes GenericModel typedef

function naiveBayes(X,y)
	# Implementation of naive Bayes classifier for binary features

	(n,d) = size(X)

  # Compute number of classes, assuming y in {1,2,...,k}
  k = maximum(y)

  # We will store p(y(i) = c) in p_y(c)
  counts = zeros(k)
  for i in 1:n
    counts[y[i]] += 1
  end
  p_y = counts ./ n


  # We will store p(x(i,j) = 1 | y(i) = c) in p_xy(1,j,c)
  # We will store p(x(i,j) = 0 | y(i) = c) in p_xy(2,j,c)
  #p_xy = (1/2)ones(2,d,k)
  p_xy = ones(2,d,k)
  countOne = 0
  countZero = 0
  #corrected p_xy
  for j in 1:d
	  for c in 1:k
		  for i in 1:n
			  if (X[i,j] == 1) && (y[i] == c)
				  countOne += 1
			  elseif (X[i,j] == 0) && (y[i] == c)
				  countZero += 1
			  end
		  end
		  p_xy[1,j,c]=(countOne/n)/p_y[c]
		  p_xy[2,j,c]=(countZero/n)/p_y[c]
		  countOne = 0
		  countZero = 0
	  end
  end
#end of corrected p_xy

  function predict(Xhat)
    (t,d) = size(Xhat)
    yhat = zeros(t)

    for i in 1:t
      # p_yx = p_y*prod(p_xy) for the appropriate x and y values
      p_yx = copy(p_y)
      for j in 1:d
        if Xhat[i,j] == 1
          for c in 1:k
            p_yx[c] *= p_xy[1,j,c]
          end
        else
          for c in 1:k
            p_yx[c] *= p_xy[2,j,c]
          end
        end
      (~,yhat[i]) = findmax(p_yx)
      end
    end
    return yhat
  end

	return GenericModel(predict)
end
