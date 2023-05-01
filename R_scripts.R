######################
#### Mantel Tests ####
######################

library(vegan)

#### Defining similarity
FCSCsim = dist(hybrid_Connectome)
FCsim = dist(FC_Connectome)
SCsim = dist(SC_Connectome)
ERsim = dist(EmoReg)

#### FCSC similarity correlated to EmoReg Similarity?
mantel(FCSCsim, ERsim, method = "spearman", permutations = 10000)

#### FC similarity or SC similarity correlated to EmoReg Similarity?
mantel(FCsim, ERsim, method = "spearman", permutations = 10000)
mantel(SCsim, ERsim, method = "spearman", permutations = 10000)

#### FCSC similarity correlated to EmoReg Similarity after controlling for FC similarity or SC similarity?
mantel.partial(FCSCsim, ERsim, FCsim, method = "spearman", permutations = 10000)
mantel.partial(FCSCsim, ERsim, SCsim, method = "spearman", permutations = 10000)


######################################
#### Principal Component Analysis ####
######################################

library(parameters)

#### PCA of Emotion Regulation Strategies across ERQ, CERQ, and COPE
pca_ER = prcomp(EmoReg, center = T, scale. = T)
summary(pca_ER)
pca_ER
plot(pca_ER, type = "l")

#### number of PCs to keep
pca_ER_comp = n_components(EmoReg ,type = "PCA", rotation = "oblimin")
summary(pca_ER_comp)
pca_ER_comp

#### resultant PCs
pca_ER_1 = pca_ER$x[,1]
pca_ER_2 = pca_ER$x[,2]
pca_ER_3 = pca_ER$x[,3]

#### repeated for CCSC questionnaire from the HBN data
pca_ER_HBN = prcomp(EmoReg_HBN, center = T, scale. = T)
summary(pca_ER_HBN)
pca_ER_HBN
plot(pca_ER_HBN, type = "l")
pca_ER_comp_HBN = n_components(EmoReg_HBN ,type = "PCA", rotation = "oblimin")
summary(pca_ER_comp_HBN)
pca_ER_comp_HBN
pca_ER_HBN_1 = pca_ER_HBN$x[,1]


############################
#### Mediation Analysis ####
############################

library(mediation)

#### defining dataframe
MedDF = cbind(pca_ER_HBN_1, ERnetwork_sum, PA)

#### linear models
mdl1=lm(pca_ER_HBN_1 ~ ERnetwork_sum, MedDF)
summary(mdl1)
mdl2=lm(PA ~ pca_ER_HBN_1 + ERnetwork_sum, MedDF)
summary(mdl2)
mdl3=lm(PA ~ ERnetwork_sum, MedDF)
summary(mdl3)
mdl4=lm(PA ~ pca_ER_HBN_1, MedDF)
summary(mdl4)

#### mediation
results = mediation::mediate(mdl1, mdl2, treat='ERnetwork_sum', mediator='pca_ER_HBN_1', boot = T, sims=5000)
summary(results)


