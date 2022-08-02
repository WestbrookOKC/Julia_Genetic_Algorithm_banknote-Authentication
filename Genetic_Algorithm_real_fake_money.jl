using CSV, DataFrames,Flux,Plots,Random,Distributions
real_money = DataFrame(CSV.File("/Users/fengkaiqi/Desktop/real-notreal-money/real-money-traindata.csv",normalizenames=true))#533*5
fack_money =DataFrame(CSV.File("/Users/fengkaiqi/Desktop/real-notreal-money/notgood-traindata.csv",normalizenames=true))#427*5
x_real_money = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(real_money) ]#533
x_fake_money = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(fack_money) ]#427
xs = vcat(x_real_money,x_fake_money)
using Flux:onehot
ys = vcat( fill(onehot(0,0:1), size(x_real_money)),
           fill(onehot(1,0:1), size(x_fake_money)))
           
real_money_test = DataFrame(CSV.File("/Users/fengkaiqi/Desktop/real-notreal-money/real-money-testdata.csv",normalizenames=true))
fack_money_test =DataFrame(CSV.File("/Users/fengkaiqi/Desktop/real-notreal-money/notgood-testdata.csv",normalizenames=true))
x_real_money_test = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(real_money_test) ]#229
x_fake_money_test = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(fack_money_test) ]#183
xss = vcat(x_real_money_test,x_fake_money_test)
using Flux:onehot
yss = vcat( fill(onehot(0,0:1), size(x_real_money_test)),
           fill(onehot(1,0:1), size(x_fake_money_test)))
testbatch = (Flux.batch(xss), Flux.batch(yss))
         
using Flux
layer1 = Dense(4,8,σ)
layer2 = Dense(8,2,identity)
using Statistics
using Printf

Actual = [fill(0,size(x_real_money)); fill(1,size(x_fake_money))]
#prediction(i) = findmax(model(Flux.batch(xs[i])))[2] - 1
function Confusion_Martix!(model)
    TP,FP,TN,FN = 0,0,0,0
    for i in 1:960
        predict_value = findmax(model(Flux.batch(xs[i])))[2] - 1
        if predict_value == 0 && Actual[i] ==  0
            TP += 1
        end
        if predict_value == 0 && Actual[i] ==  1
            FP += 1
        end
        if predict_value == 1 && Actual[i] ==  1
            TN += 1
        end
        if predict_value == 1 && Actual[i] ==  0
            FN += 1
        end    
    end
    print([TP FN
            FP TN])
end
function getmodel()
    model = Chain(Dense(4,8,σ),Dense(8,2,identity),softmax)
end
function population(size)
    list = []
    for i in 1:size
        rand_model = getmodel()
        append!(list,[rand_model])
    end
    return list
end
function accuracy_each_model(rand_model)
    Actual = [fill(0,size(x_real_money)); fill(1,size(x_fake_money))]
    prediction(i) = findmax(rand_model(xs[i]))[2] - 1
    TP,FP,TN,FN = 0,0,0,0
    for i in 1:960
        predict_value = prediction(i)
        if predict_value == 0 && Actual[i] ==  0
            TP += 1
        end
        if predict_value == 0 && Actual[i] ==  1
            FP += 1
        end
        if predict_value == 1 && Actual[i] ==  1
            TN += 1
        end
        if predict_value == 1 && Actual[i] ==  0
            FN += 1
        end    
    end
    accuracy = (TN + TP) / (TP+TN+FN+FP)
    return accuracy
end

function model_list_accuracy()
    model_list = population(100)
    dict = Dict()
    acc = []
    for model in model_list
        key = accuracy_each_model(model)
        append!(acc,key)
        value = model
        push!(dict,key =>value)
    end
    x = sort(acc,rev = true)
    final_list = zeros(60)
    for i in 1:60
        final_list[i] = x[i]
    end
    model1_list = []
    for i in 1:60
        append!(model1_list,dict[x[i]])
    end
    father = dict[final_list[1]]
    mother = dict[final_list[2]]
    return father, mother,final_list,model1_list
end
a = model_list_accuracy()
father = a[1]
mother = a[2]
final_list = a[3]
xxx= a[4]




layer = 2
Crossover_rate= 0.05
#Do Crossover here 
function crossover(father,mother)
    nn=deepcopy(father)
    for i in 1:layer
        select_one_layer = shuffle(collect(1:layer))[1]
        mother_bias=mother[select_one_layer].bias
        picked_one_row=shuffle(collect(1:size(mother_bias)[1]))[1]
        if rand() < Crossover_rate
            nn[select_one_layer].bias[picked_one_row]=mother_bias[picked_one_row]
        end
    end
    
    for i in 1:layer
        select_one_layer=shuffle(collect(1:layer))[1]
        mother_weight=mother[select_one_layer].weight
        picked_one_row=shuffle(collect(1:size(mother_weight)[1]))[1]
        picked_one_col=shuffle(collect(1:size(mother_weight)[2]))[1]
        if rand() < Crossover_rate
            nn[select_one_layer].weight[picked_one_row,picked_one_col]=mother_weight[picked_one_row,picked_one_col]
        end
    end
    return nn
end
new_child = crossover(father,mother)
Mutation_rate = 0.01
#Do mutation here
function mutation(new_child)
    nn=deepcopy(new_child)
    for i in 1:layer
        select_one_layer = shuffle(collect(1:layer))[1]
        bias=nn[select_one_layer].bias
        picked_one_row=shuffle(collect(1:size(bias)[1]))[1]
        if rand() < Mutation_rate
            nn[select_one_layer].bias[picked_one_row]+=rand(Uniform(-0.5,0.5),1)[1]
        end
    end
    
    for i in 1:layer
        select_one_layer=shuffle(collect(1:layer))[1]
        weight=nn[select_one_layer].weight
        picked_one_row=shuffle(collect(1:size(weight)[1]))[1]
        picked_one_col=shuffle(collect(1:size(weight)[2]))[1]
        if rand() < Mutation_rate
            nn[select_one_layer].weight[picked_one_row,picked_one_col]+=rand(Uniform(-0.5,0.5),1)[1]
        end
    end
    return nn
end
mutation(new_child)

function fitness()
    y = []
    for i in 1:40
        a = model_list_accuracy()
        father = a[1]
        mother = a[2]
        orginal_list = a[3]
        append!(y,orginal_list)
        new_child = crossover(father,mother)
        final_child = mutation(new_child)
        x = accuracy_each_model(final_child)
        append!(y,x)
    end
    z = sort(y,rev = true)
    acc = zeros(100)
    for i in 1:100
        acc[i] = z[i]
    end
    mean_acc = mean(acc)
    return mean_acc
end
accuracy_arr = []
for i in 1:2
    x = fitness()
    append!(accuracy_arr,x)
end
xxxx = sort(accuracy_arr)
function show_accuracy()
    plot!(1:length(xxxx), xxxx, label = "accuracy")
    plot!(xlabel = "Iteration")
end






Confusion_Martix!(father)

show_accuracy()
