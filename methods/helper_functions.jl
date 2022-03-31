using Statistics

function compute_tanh(x)
    val = tanh.(x)
    return val
end

function compute_tanh′(x)
    val = 1 - tanh.(x)^2
    return val
end

function mean_squared_error(y_true, y_pred)
    val = mean!([1.], (y_true-y_pred)^2) # TO DO: fix mean!
    return val
end

function mean_squared_error′(y_true, y_pred)
    val = 2*(y_pred-y_true)/y_true.size
    return val
end