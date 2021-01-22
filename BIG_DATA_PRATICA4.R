# Big Data na PrÃ¡tica 4 - Customer Churn Analytics 
#Script desenvolvido no curso Formação Cientista de Dados da DataScienci Academy
#https://www.datascienceacademy.com.br/bundles?bundle_id=formacao-cientista-de-dados

# A rotatividade (churn) de clientes ocorre quando clientes ou assinantes param de fazer negÃ³cios 
# com uma empresa ou serviÃ§o. TambÃ©m Ã© conhecido como perda de clientes ou taxa de cancelamento.

# Um setor no qual saber e prever as taxas de cancelamento é particularmente útil é o setor de telecomunicaÃ§Ãµes, 
# porque a maioria dos clientes tem vÃ¡rias opÃ§Ãµes de escolha dentro de uma localizaÃ§Ã£o geogrÃ¡fica.

# Neste projeto, vamos prever a rotatividade (churn) de clientes usando um conjunto de dados de telecomunicaÃ§Ãµes. 
# Usaremos a regressão logística, a Ã¡rvore de decisÃ£o e a floresta aleatÃ³ria como modelos de Machine Learning. 

# Usaremos um dataset oferecido gratuitamente no portal IBM Sample Data Sets. 
# Cada linha representa um cliente e cada coluna contÃ©m os atributos desse cliente.

# https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/

#Carlos E. Carvalho
#carlos.e.carvalho@gmail.com
#LinkedIn: https://www.linkedin.com/in/carlos-carvalho-93204b13/

# Definindo o diretÃ³rio de trabalho
setwd("D:/CIENTISTA_DADOS/1_BIG_DATA_R_AZURE/CAP06")
getwd()


# Carregando os pacotes
library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)
library(e1071)

tinytex::install_tinytex()


##### Carregando e Limpando os Dados ##### 

# Os dados brutos contÃ©m 7043 linhas (clientes) e 21 colunas (recursos). 
# A coluna "Churn" é o nosso alvo.
churn <- read.csv("Telco-Customer-Churn.csv")
View(churn)

# Usamos sapply para verificar o número de valores ausentes (missing) em cada coluna. 
# Descobrimos que há 11 valores ausentes nas colunas "TotalCharges". 
# EntÃ£o, vamos remover todas as linhas com valores ausentes.
sapply(churn, function(x) sum(is.na(x)))
complete.cases(churn)
?complete.cases
#Coloca no dataframe churn apenas os usuários que tem todas as informações completas
churn <- churn[complete.cases(churn), ]
sapply(churn, function(x) sum(is.na(x)))

# Olhe para as variÃ¡veis, podemos ver que temos algumas limpezas e ajustes para fazer.
# 1. Vamos mudar "No internet service" para "No" por seis colunas, que são: 
# "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "streamingTV", 
# "streamingMovies".
#Isso é limpeza dos dados. Decisão do cientista de dados
cols_recode1 <- c(10:15)  #Vetor com os números inteiros entre 10 e 15, representa as colunas que serão modificadas
for(i in 1:ncol(churn[,cols_recode1])){
  print(i)
  churn[,cols_recode1][,i] <- as.factor(mapvalues(churn[,cols_recode1][,i], from = c("No internet service"), to = c("No")))
}

# 2. Vamos mudar "No phone service" para "No" para a coluna MultipleLines


View(churn[,cols_recode1]) #Cria um dataframe com todas as linhas originais e as colunas 10,11,12,13,14 e 15

# 3. Como a permanÃªncia mínima é de 1 mês e a permanência máxima é de 72 meses, 
# podemos agrupá-los em cinco grupos de posse (tenure): 
# â 0-12 Mês â, â12â24 MÃªsâ, â24â48 Mesesâ, â48â60 MÃªsâ MÃªs â,â> 60 MÃªsâ
min(churn$tenure); max(churn$tenure)
group_tenure <- function(tenure){
  if(tenure >= 0 & tenure <= 12){
    return("0 - 12 Month")
  }else if(tenure > 12 & tenure <= 24){
    return("12 - 24 Months")
  }else if(tenure > 24 & tenure <= 48){
    return("24 - 48 Months")
  }else if(tenure > 48 & tenure <= 60){
    return("48 - 60 Months")
  }else if(tenure > 60){
    return("> 60 Months")
  }
}

#Aplica a função criada acima, criando uma coluna chama tenure_group
churn$tenure_group <- sapply(churn$tenure, group_tenure)
#Transforma os valores da nova coluna para fator
churn$tenure_group <- as.factor(churn$tenure_group)

#A coluna SeniorCitizen é a única que está com 0 ou 1.  As outras estão com No ou Yes.
#Então vamos alterar essa coluna para deixar todas iguais.
churn$SeniorCitizen <- as.factor(mapvalues(churn$SeniorCitizen, from = c("0","1"), to = c("No", "Yes")))

# 5. Removemos as colunas que nãoo precisamos para a análise.
churn$customerID <- NULL
churn$tenure <- NULL


############### ANÁLISE EXPLORATÓRIA DE DADOS E SELEÇÃO DE RECURSOS ############################

#Observar a correlação entre as variáveis numéricas
numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
corr.matrix
corrplot(corr.matrix, main = "\n\nGráfico de correlação para variáeis numéricas", method = "number")

#Gráficos de barra de variáveis categóricas
p1 <- ggplot(churn, aes(x=gender)) + ggtitle("Gender") + xlab("sexo") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_excel()
p2 <- ggplot(churn, aes(x=SeniorCitizen)) + ggtitle("Senior Citizen") + xlab("Senior Citizen") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_excel()
p3 <- ggplot(churn, aes(x = Partner)) + ggtitle("Partner") + xlab("Parceiros") + 
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_excel()
p4 <- ggplot(churn, aes(x = Dependents)) + ggtitle("Dependents") + xlab("Dependentes") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_excel()
grid.arrange(p1, p2, p3, p4, ncol = 2)

p5 <- ggplot(churn, aes(x=PhoneService)) + ggtitle("Phone Service") + xlab("Telefonia") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_classic()
p6 <- ggplot(churn, aes(x=MultipleLines)) + ggtitle("MultipleLines") + xlab("Linhas Multiplas") + 
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_classic()
p7 <- ggplot(churn, aes(x = InternetService)) + ggtitle("Internet Service") + xlab("Serviço de Internet") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_classic()
p8 <- ggplot(churn, aes(x=OnlineSecurity)) + ggtitle("Online Security") + xlab("Segurança") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = ) + ylab("Percentual") + coord_flip() + theme_classic()
grid.arrange(p5, p6, p7, p8, ncol = 2)

p9 <- ggplot(churn, aes(x=OnlineBackup)) + ggtitle("Backup Online") + xlab("Backup Online") + 
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_light()
p10 <- ggplot(churn, aes(x=DeviceProtection)) + ggtitle("Device Protection") + xlab("Proteção") + 
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_light()
p11 <- ggplot(churn, aes(x=TechSupport)) + ggtitle("Tech Support") + xlab("Suporte Técnico") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_light()
p12 <- ggplot(churn, aes(x=StreamingTV)) + ggtitle("Streaming TV") + xlab("Streaming TV") + 
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_light()
grid.arrange(p9, p10, p11, p12, ncol = 2)

p13 <- ggplot(churn, aes(x=StreamingMovies)) + ggtitle("Streaming Movies") + xlab("Streaming Movies") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_economist()
p14 <- ggplot(churn, aes(x=Contract)) + ggtitle("Contract") + xlab("Contratos") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_economist()
p15 <- ggplot(churn, aes(x=PaperlessBilling)) + ggtitle("Paperless Billing") + xlab("Conta Virtual") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_economist()
p16 <- ggplot(churn, aes(x=PaymentMethod)) + ggtitle("Payment Method") + xlab("Método de Pagamento") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_economist()
p17 <- ggplot(churn, aes(x=tenure_group)) + ggtitle("Tempo de Contrato") + xlab("Tempo de Contrato") +
  geom_bar(aes(y=100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentual") + coord_flip() + theme_economist()
grid.arrange(p13, p14, p15, p16, p17, ncol = 2)

# Todas as variÃ¡veis categóricas parecem ter uma distribuiÃ§Ã£o razoavelmente ampla, 
# portanto, todas elas serÃ£o mantidas para anÃ¡lise posterior.

################### MODELAGEM PREDITIVA #########################

#Regressão Logística

#Primeiro dividimos os dados em conjuntos de treinamento e teste
intrain <- createDataPartition(churn$Churn, p=0.7, list = FALSE)
set.seed(2017)
intrain
training <- churn[intrain,]
testing <- churn[-intrain,]

#Verificar se a divisão está correta
dim(training); dim(testing)

#Treinamento do modelo de regressão logística
#Fitting do modelo
LogModel <- glm(Churn ~ ., family=binomial(link = "logit"), data = training)
print(summary(LogModel))

#Análise de variância - ANOVA
?anova
anova(LogModel, test = "Chisq")

# Analisando a tabela de variância, podemos ver a queda no desvio ao adicionar cada variÃ¡vel 
# uma de cada vez. Adicionar InternetService, Contract e OnlineSecurity reduz 
# significativamente o desvio residual. 
# As outras variÃ¡veis, como PaymentMethod e Dependents, parecem melhorar menos o modelo, 
# embora todos tenham valores p baixos.

testing$Churn <- as.character(testing$Churn)
testing$Churn[testing$Churn == "No"] <- "0"
testing$Churn[testing$Churn == "Yes"] <- "1"
fitted.results <- predict(LogModel, newdata = testing, type = "response")
fitted.results <- ifelse(fitted.results > 0.5,1,0)
fitted.results
misClassificError <- mean(fitted.results != testing$Churn)
print(paste("Taxa de acerto da regressão logística: ", 1-misClassificError))

#Matriz de confusão de regressão logística
print("Confusion Matrix para Regressão Logística"); table(testing$Churn, fitted.results > 0.5)

#Odds Ratio
#Uma das medidas desemplenho interessantes na Regressão Logística é o Odds Ratio
#Basicamente, a Odds Ratio é a chance de um evento acontecer
exp(cbind(OR = coef(LogModel), confint(LogModel)))

################### Árvore de Decisão
# VisualizaÃ§Ã£o da Ãrvore de DecisÃ£o
# Para fins de ilustração, vamos usar apenas três variáveis para plotar 
# Árvores de decisão, elas são Contrato, tenure_group e PaperlessBilling.

tree <- ctree(Churn ~ Contract + tenure_group + PaperlessBilling, training)
plot(tree, type = "simple")

tree <- ctree(Churn ~ Contract + MonthlyCharges + PaperlessBilling, training)
plot(tree, type = "simple")
tree

# Matriz de ConfusÃ£o da Ãrvore de DecisÃ£o
# Estamos usando todas as variÃ¡veis para tabela de matriz de confusÃ£o de produto e fazer previsÃµes.
pred_tree <- predict(tree, testing)
print("Confusion Matrix para Decision Tree"); table(Predicted = pred_tree, Actual = testing$Churn)

#Precisão da árvore de decisão
p1 <- predict(tree, training)
tab1 <- table(Predicted=p1, Actual = training$Churn)
tab2 <- table(Predicted = pred_tree, Actual = testing$Churn)

print(paste("Decision Tree Accuracy", sum(diag(tab2))/sum(tab2)))

########## Random Forest ###################
set.seed(2017)
rfModel <- randomForest(Churn ~ ., data = training)
print(rfModel)
plot(rfModel)

#Prevendo valores com dados de teste
pred_rf <- predict(rfModel, testing)

#Confusion Matrix
print("Confusion Matrix para Random Forest"); table(testing$Churn, pred_rf)

#Recursos Mais Importantes
varImpPlot(rfModel, sort = T, n.var = 10, main="Top 10 Feature Importance")

















