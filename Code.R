train = read.csv('C:/Users/Vittorio/Desktop/uni/A_stat/altro uni/sus/train.csv')
bareme_tab = read.table('C:/Users/Vittorio/Desktop/uni/A_stat/altro uni/sus/bareme_tab.txt',header=TRUE)

# Creo variabile basata sulla Bareme table
cust = train$Bareme_table_code_customer
oth = train$Bareme_table_code_other_driver

resp = paste(cust,oth,sep='-')
resp = factor(resp)
summary(resp)
levels(resp) = c(levels(resp),'Other')
resp[resp %in% levels(resp)[which(table(resp)<500)]] = 'Other'
resp = factor(resp)
train$resp = resp

table(cust)
table(oth)
n = nrow(train)
fault = rep(NA,n)
for (i in 1:n) {
  y = train$Bareme_table_code_customer[i]
  x = train$Bareme_table_code_other_driver[i]
  fault[i] = as.character(bareme_tab[rownames(bareme_tab)==y,rownames(bareme_tab)==x])
}
fault = factor(fault)
train$fault = fault

# Tolgo weight
train = train[,-1]

# Differenza date
accident = as.POSIXct(as.character(train$Date_of_accident),format="%Y-%m-%d")
opening = as.POSIXct(as.character(train$Date_claim_opening),format="%Y-%m-%d")
diff_date = as.numeric(difftime(opening,accident,units='days'))
train$diff_date = diff_date
# Tolgo le date
train = train[,-c(2,3)]

# Customer_code
# Ok

# Customer merit class
merit = train$Customer_merit_class
summary(merit)
# Imputo con mediana
merit[is.na(merit)] = 1
merit[merit>1] = 0
train$Customer_merit_class = factor(merit)

# Virtuous driver
flag = train$Flag_virtuous_driver
summary(flag)
table(flag)
train = train[,-6]

# Black box
train$Presence_of_black_box = factor(train$Presence_of_black_box)
summary(train$Presence_of_black_box)

# Tolgo Bareme
train = train[,-c(7,8)]

# Provincia
summary(train$Province_code)
train = train[,-7]

# Regione
summary(train$Region)
nord = c("VALLE D'AOSTA","VENETO","LOMBARDIA","PIEMONTE","EMILIA-ROMAGNA",
         "TRENTINO-ALTO ADIGE","FRIULI-VENEZIA GIULIA","LIGURIA")
centro = c("TOSCANA","LAZIO","MARCHE","UMBRIA")
sud = c("SICILIA","PUGLIA","ABRUZZO","CAMPANIA","MOLISE","CALABRIA","SARDEGNA","BASILICATA")            
zona = rep(NA,n)
zona[train$Region %in% nord] = 'NORD'
zona[train$Region %in% centro] = 'CENTRO'
zona[train$Region %in% sud] = 'SUD'
train$zona = factor(zona)
train = train[,-7]

# Data immatricolazione
immatr = as.POSIXct(as.character(train$Date_vehicle_immatriculation),format="%Y-%m-%d")
acc_immatr = as.numeric(difftime(accident,immatr,units='days'))
summary(acc_immatr)
acc_immatr[is.na(acc_immatr)] = mean(acc_immatr, na.rm=TRUE)
train$acc_immatr = acc_immatr
train = train[,-7]

# Tipologia veicolo
summary(train$Vehicle_typology)
levels(train$Vehicle_typology) = c(levels(train$Vehicle_typology),'Other')
train$Vehicle_typology[train$Vehicle_typology %in% c("Coupe'","Off-road","Open","Minivan")] = 'Other'
train$Vehicle_typology[is.na(train$Vehicle_typology)] = 'Sedan'
train$Vehicle_typology = factor(train$Vehicle_typology)

# Power
summary(train$Vehicle_power_source)
levels(train$Vehicle_power_source) = c(levels(train$Vehicle_power_source),'Other')
train$Vehicle_power_source[train$Vehicle_power_source %in% c("Electric","Hybrid / Diesel",
                                                             "Hybrid / Petrol","LPG","Methane",
                                                             "Petrol / LPG", "Petrol / methane")] = 'Other'
train$Vehicle_power_source[is.na(train$Vehicle_power_source)] = 'Other'
train$Vehicle_power_source = factor(train$Vehicle_power_source)

# Catalitic
summary(train$Vehicle_catalytic_converter)
train = train[,-9]

# Flag Hybryd
train = train[,-9]

# Annaul Distance
train = train[,-9]

# History
train = train[,-9]

# Engine capacity
summary(train$Vehicle_engine_capacity)
train$Vehicle_engine_capacity[is.na(train$Vehicle_engine_capacity)] = mean(na.omit(train$Vehicle_engine_capacity))

# RPM engine
summary(train$Vehicle_engine_rpm)
train$Vehicle_engine_rpm[is.na(train$Vehicle_engine_rpm)] = mean(na.omit(train$Vehicle_engine_rpm))

# KGM
# Ok

# RPM
summary(train$Vehicle_torque_rpm)
train$Vehicle_torque_rpm[is.na(train$Vehicle_torque_rpm)] = mean(na.omit(train$Vehicle_torque_rpm))

# CYL
train = train[,-13]

# HP
summary(train$Vehicle_fiscal_HP)
train$Vehicle_fiscal_HP[is.na(train$Vehicle_fiscal_HP)] = mean(na.omit(train$Vehicle_fiscal_HP))

# Price
summary(train$Vehicle_price_euro)
train$Vehicle_price_euro[is.na(train$Vehicle_price_euro)] = mean(na.omit(train$Vehicle_price_euro))

# Speed
summary(train$Vehicle_max_speed_kmh)
train$Vehicle_max_speed_kmh[is.na(train$Vehicle_max_speed_kmh)] = mean(na.omit(train$Vehicle_max_speed_kmh))

# Acceleration
train = train[,-16]

# Full load
train = train[,-20]

# Supercharging
train = train[,-22]

# Target in log
train$ylog = log(train$Target_cost_euro)


# Stima-Validazione
caso = sample(1:n,1694) # 75%
sss = train[-caso,]
vvv = train[caso,]


### START ANALYSIS ###
# Random Forest
library(quantregForest)
cb1 = sample(1:nrow(sss),floor(nrow(sss)/2)) 
cb2 = setdiff(1:nrow(sss),cb1)

rf = quantregForest(sss[cb1,-c(1,27)], sss[cb1,27], nthreads=6, nodesize=5, mtry=5,
                   ntree=300,xtest=sss[cb2,-c(1,27)], ytest=sss[cb2,27], 
                   keep.forest=T)
plot(rf, main='Errore nel validation set')
varImpPlot(rf)
p.rf = predict(rf, newdata=vvv)
mae.rf = mean(abs(exp(p.rf[,2])-vvv$Target_cost_euro))
mae.rf

# Boosting
require(Matrix)
require(caret)
require(xgboost)
train_X = sparse.model.matrix(Target_cost_euro~.-1,data=sss[cb1,-27])
train_y = sss$ylog[cb1] #sss$Target_cost_euro[cb1]
test_X = sparse.model.matrix(Target_cost_euro~.-1,data=sss[cb2,-27])
test_y = sss$ylog[cb2] #sss$Target_cost_euro[cb2]

train_X = sparse.model.matrix(Target_cost_euro~.-1,data=sss[,-27])
train_y = sss$ylog #sss$Target_cost_euro[cb1]
test_X = sparse.model.matrix(Target_cost_euro~.-1,data=vvv[,-27])
test_y = vvv$ylog #sss$Target_cost_euro[cb2]

train_X_xgb = xgb.DMatrix(train_X,label = train_y)
test_X_xgb = xgb.DMatrix(test_X,label = test_y)
my_watchlist = list(train = train_X_xgb,test = test_X_xgb)

loss_huber_xgb = function(preds,dtrain){
  labels = getinfo(dtrain, "label")
  x = preds - labels
  h = 1
  scale = 1+(x/h)^2
  grad = x / sqrt(1+(x/h)^2)
  hess = 1 / (1+(x/h)^2)^(1.5)
  return(list(grad = grad, hess = hess))
}

param = list(max_depth = 4, # profondità alberi
             eta = 0.01) # learning rate

bst = xgb.train(data = train_X_xgb, params = param,
                 early_stopping_rounds = 100,
                 objective = loss_huber_xgb,
                 nrounds = 10^4, watchlist = my_watchlist,
                 method = "xgbTree", eval_metric = 'mae')

v_x = sparse.model.matrix(Target_cost_euro~.-1,data=vvv[,-27])

p.bost = predict(bst, newdata=v_x)
mae.bost = mean(abs(exp(p.bost)-vvv$Target_cost_euro))
mae.bost

# Test
test = read.csv('test.csv')
# Creo variabile basata sulla Bareme table
cust = test$Bareme_table_code_customer
oth = test$Bareme_table_code_other_driver

resp = paste(cust,oth,sep='-')
resp = factor(resp)
summary(resp)
levels(resp) = c(levels(resp),'Other')
resp[resp %in% levels(resp)[which(table(resp)<160)]] = 'Other'
resp = factor(resp)
test$resp = resp

n = nrow(test)
fault = rep(NA,n)
for (i in 1:n) {
  y = test$Bareme_table_code_customer[i]
  x = test$Bareme_table_code_other_driver[i]
  fault[i] = as.character(bareme_tab[rownames(bareme_tab)==y,rownames(bareme_tab)==x])
}
fault = factor(fault)
test$fault = fault

# Differenza date
accident = as.POSIXct(as.character(test$Date_of_accident),format="%Y-%m-%d")
opening = as.POSIXct(as.character(test$Date_claim_opening),format="%Y-%m-%d")
diff_date = as.numeric(difftime(opening,accident,units='days'))
test$diff_date = diff_date
# Tolgo le date
test = test[,-c(1,2)]

# Customer_code
levels(test$Customer_code3) = c(levels(test$Customer_code3),"PR")
provv = c(as.character(test$Customer_code3),"PR")
provv = factor(provv)
provv = provv[-which(provv=="PR")]
test$Customer_code3 = provv
levels(test$Customer_code3)
summary(test$Customer_code3)

# Customer merit class
merit = test$Customer_merit_class
summary(merit)
# Imputo con mediana
merit[is.na(merit)] = 1
merit[merit>1] = 0
merit = factor(merit)
test$Customer_merit_class = merit

# Virtuous driver
flag = test$Flag_virtuous_driver
summary(flag)
table(flag)
boxplot(merit~flag) # Poco informativa
test = test[,-5]

# Black box
test$Presence_of_black_box = factor(test$Presence_of_black_box)
summary(test$Presence_of_black_box)

# Tolgo Bareme
test = test[,-c(6,7)]

# Provincia
summary(test$Province_code)
test = test[,-6]

# Regione
summary(test$Region)
nord = c("VALLE D'AOSTA","VENETO","LOMBARDIA","PIEMONTE","EMILIA-ROMAGNA",
         "TRENTINO-ALTO ADIGE","FRIULI-VENEZIA GIULIA","LIGURIA")
centro = c("TOSCANA","LAZIO","MARCHE","UMBRIA")
sud = c("SICILIA","PUGLIA","ABRUZZO","CAMPANIA","MOLISE","CALABRIA","SARDEGNA","BASILICATA")            
zona = rep(NA,n)
zona[test$Region %in% nord] = 'NORD'
zona[test$Region %in% centro] = 'CENTRO'
zona[test$Region %in% sud] = 'SUD'
test$zona = factor(zona)
test = test[,-6]

# Data immatricolazione
immatr = as.POSIXct(as.character(test$Date_vehicle_immatriculation),format="%Y-%m-%d")
acc_immatr = as.numeric(difftime(accident,immatr,units='days'))
summary(acc_immatr)
acc_immatr[is.na(acc_immatr)] = mean(acc_immatr, na.rm=TRUE)
test$acc_immatr = acc_immatr
test = test[,-6]

# Tipologia veicolo
summary(test$Vehicle_typology)
levels(test$Vehicle_typology) = c(levels(test$Vehicle_typology),'Other')
test$Vehicle_typology[test$Vehicle_typology %in% c("Coupe'","Off-road","Open","Minivan")] = 'Other'
test$Vehicle_typology[is.na(test$Vehicle_typology)] = 'Sedan'
test$Vehicle_typology = factor(test$Vehicle_typology)

# Power
summary(test$Vehicle_power_source)
levels(test$Vehicle_power_source) = c(levels(test$Vehicle_power_source),'Other')
test$Vehicle_power_source[test$Vehicle_power_source %in% c("Electric","Hybrid / Diesel",
                                                             "Hybrid / Petrol","LPG","Methane",
                                                             "Petrol / LPG", "Petrol / methane")] = 'Other'
test$Vehicle_power_source[is.na(test$Vehicle_power_source)] = 'Other'
test$Vehicle_power_source = factor(test$Vehicle_power_source)

# Catalitic
summary(test$Vehicle_catalytic_converter)
test = test[,-8]

# Flag Hybryd
test = test[,-8]

# Annaul Distance
test = test[,-8]

# History
test = test[,-8]

# Engine capacity
summary(test$Vehicle_engine_capacity)
test$Vehicle_engine_capacity[is.na(test$Vehicle_engine_capacity)] = mean(na.omit(test$Vehicle_engine_capacity))

# RPM engine
summary(test$Vehicle_engine_rpm)
test$Vehicle_engine_rpm[is.na(test$Vehicle_engine_rpm)] = mean(na.omit(test$Vehicle_engine_rpm))

# KGM
# Ok

# RPM
summary(test$Vehicle_torque_rpm)
test$Vehicle_torque_rpm[is.na(test$Vehicle_torque_rpm)] = mean(na.omit(test$Vehicle_torque_rpm))

# CYL
test = test[,-12]

# HP
summary(test$Vehicle_fiscal_HP)
test$Vehicle_fiscal_HP[is.na(test$Vehicle_fiscal_HP)] = mean(na.omit(test$Vehicle_fiscal_HP))

# Price
summary(test$Vehicle_price_euro)
test$Vehicle_price_euro[is.na(test$Vehicle_price_euro)] = mean(na.omit(test$Vehicle_price_euro))

# Speed
summary(test$Vehicle_max_speed_kmh)
test$Vehicle_max_speed_kmh[is.na(test$Vehicle_max_speed_kmh)] = mean(na.omit(test$Vehicle_max_speed_kmh))

# Acceleration
test = test[,-15]

# Full load
test = test[,-19]

# Supercharging
test = test[,-21]

# PREDICTION
yhat = predict(rf, newdata=test)
yhat = exp(yhat[,2])
write.table(yhat,'quantregForest1.txt',row.names=FALSE,col.names=FALSE)

testx = sparse.model.matrix(~.-1,data=test)
yhat = predict(bst, testx)
yhat = exp(yhat)
write.table(yhat,'bst1.txt',row.names=FALSE,col.names=FALSE)
