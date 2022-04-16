using Statistics

function compute_tanh(x)
    val = tanh.(x)
    return val
end

function compute_tanh′(x)
    val = tanh.(x).^2 .* (-1) .+ 1
    return val
end

function mean_squared_error(y_true, y_pred)
    val = first(mean((y_true.-y_pred).^2, dims=2))
    return val
end

function mean_squared_error′(y_true, y_pred)
    val = 2*(y_pred.-y_true)/length(y_true)
    return val
end

