# using R 4.3
# the beginning of this script is almost identical to the protocol described at:
# https://star-protocols.cell.com/protocols/1733
library(data.table)
library(dplyr)
library(ukbpheno)
library(ggplot2)
library(ggforce)
library(tableone)
library(survminer)
library("MatchIt")

# the directory with datafiles
ukb_dir <-"33127/"

# main dataset 
fukbtab <- paste(ukb_dir,"ukb52389.tab",sep="")

# meta data file
fhtml <- paste(ukb_dir,"ukb52389.html",sep="")

# hospital inpatient data
fhesin <- paste(ukb_dir,"medical_records/hesin/hesin.txt",sep="")
fhesin_diag <- paste(ukb_dir,"medical_records/hesin/hesin_diag.txt",sep="")
fhesin_oper <- paste(ukb_dir,"medical_records/hesin/hesin_oper.txt",sep="")

# GP data
fgp_clinical <- paste(ukb_dir,"medical_records/gp/gp_clinical.txt",sep="")
fgp_scripts <- paste(ukb_dir,"medical_records/gp/gp_scripts.txt",sep="")

# Death data
fdeath_portal <- paste(ukb_dir,"medical_records/death/death.txt",sep="")
fdeath_cause_portal <- paste(ukb_dir,"medical_records/death/death_cause.txt",sep="")

# Withdrawal data
f_withdrawal <- paste(ukb_dir, "tteCurated/data/covars/withdraw33127_250_20230426.txt",sep="")

# definition table included in the package
extdata_dir <- system.file("extdata", package="ukbpheno")
fdefinitions <- system.file("extdata", "definitions_cardiometabolic_traits.tsv", package="ukbpheno")

# data setting file included in the package
fdata_setting <- system.file("extdata", "data.settings.tsv", package="ukbpheno")
dfData.settings <- fread(fdata_setting)

# process the definition table based on data setting
dfDefinitions_processed_expanded <- read_definition_table(fdefinitions, fdata_setting, extdata_dir)

# harmonize data based on cardiometabolic traits
lst.harmonized.data<-harmonize_ukb_data(f.ukbtab = fukbtab,f.html = fhtml,dfDefinitions=dfDefinitions_processed_expanded,f.gp_clinical = fgp_clinical,f.gp_scripts = fgp_scripts,f.hesin = fhesin,f.hesin_diag = fhesin_diag,f.hesin_oper =fhesin_oper,f.death_portal = fdeath_portal,f.death_cause_portal = fdeath_cause_portal,f.withdrawal_list=f_withdrawal,allow_missing_fields = TRUE)

# the authors used the baseline visit as the reference date
# we however will use the date of birth as the reference date
# reference date set by original authors -- baseline visit
df_reference_dt_v0 <- lst.harmonized.data$dfukb[,c("identifier","f.53.0.0")]

#f.52.0.0 is the birth month
birth_month <- lst.harmonized.data$dfukb[,c("identifier","f.52.0.0")]
#f.34.0.0 is the birth year
birth_year <- lst.harmonized.data$dfukb[,c("identifier","f.34.0.0")]
birth_year$f.34.0.0 <- as.character(birth_year$f.34.0.0)
birth = merge(birth_month, birth_year, by = "identifier")

#As the day is unknown, arbitrarily set everyone to the middle of the month
replace = paste0(birth$f.34.0.0,"-",birth$f.52.0.0,"-15")
replace[which(replace=="NA-NA-15")] = NA
#double checked ordering is the same (it should be)
#which(birth$identifier!=df_reference_dt_v0$identifier) returns integer(0)
df_reference_dt_v0$f.53.0.0 = replace

# establish last date for each person
death_dt<-unique(lst.harmonized.data$lst.data$tte.death.icd10.primary[,.(identifier,eventdate)])
colnames(death_dt)<-c("identifier","death")

# last assessment by center
england<-c("10003","11001","11002","11007","11008","11009","11010","11011","11012","11013","11014","11016","11017","11018","11019","11020","11021")
scotland<-c("11004","11005")
wales<-c("11003","11022", "11006","11023")

# England censor 2022-10-31
# Scotland censor 2021-07-31
# Wales censor 2018-02-28
# using a second source of the data because some genetic trait columns (kinship, etc weren't in the main data source)
temp = read.csv(paste0(ukb_dir, "phenos/21970/ukb21970.csv"))

# subset of dataset needed to clean cohort, create covariates, and censor data
censoring=subset(temp, select = c("eid", "X54.0.0", "X191.0.0", 
	"X21003.0.0", "X22001.0.0", "X22006.0.0", "X22019.0.0", "X22021.0.0"))

#awk '{ print $1, $26700, $26702, $26708, $26710, $26712, $26714, $26726 }' ../../../phenos/673008/ukb673008.tab >> algorithm_age_grepped.txt
#grep -n 'f.eid|f.42006.0.0|f.42008.0.0|f.42014.0.0|f.42016.0.0|f.42018.0.0|f.42020.0.0|f.42032.0.0' ../../../phenos/673008/ukb673008.fields.txt 

#temp2 = read.table("algorithm_age_grepped.txt",head=T)
#pheno = subset(temp2, select = c('eid', 'f.42006.0.0', 'f.42008.0.0', 'f.42014.0.0', 'f.42016.0.0', 'f.42018.0.0', 'f.42020.0.0', 'f.42032.0.0'))


# f.54.0.0 assessment center
# f.191.0.0 lost to follow-up
# Age is age at first assessment
# Sex is genetic sex
# Ethnicity is genetic ethnicity (1==Caucasian)
# SexAneuploidy indicates if person not XX or XY
# Kinship identifies people with no relatives in cohort
colnames(censoring) = c("identifier", "f.54.0.0", "f.191.0.0", 
	"Age", "Sex", "Ethnicity", "SexAneuploidy", "Kinship")

# Unrelated white Britons without aneuploidy on sex chromosome
censoring = censoring[which(censoring$Ethnicity==1 & is.na(censoring$SexAneuploidy) & censoring$Kinship==0),]

# create covars, now that we have final sample
covars = subset(censoring, select = c("identifier", "identifier", "Age", "Sex"))
colnames(covars) = c("FID", "IID", "Age", "Sex")

# these PCs were generated here: $HOME/project-UKBB/data/cal/tteCurated
# they are based in-sample EUR (~276k individuals) and using plink2 pca approx
# ~300 fewer in pcs, dropped previous round, not 100% why but not worth finding them
# followed same processing using kinship, sex aneuploidy, etc
pcs = read.table(paste0(ukb_dir, "tteCurated/data/covars/all_pcs.eigenvec"))
colnames(pcs) = c("FID", "IID", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10")
pcs$PC1 = scale(pcs$PC1)
pcs$PC2 = scale(pcs$PC2)
pcs$PC3 = scale(pcs$PC3)
pcs$PC4 = scale(pcs$PC4)
pcs$PC5 = scale(pcs$PC5)
pcs$PC6 = scale(pcs$PC6)
pcs$PC7 = scale(pcs$PC7)
pcs$PC8 = scale(pcs$PC8)
pcs$PC9 = scale(pcs$PC9)
pcs$PC10 = scale(pcs$PC10)

covars = merge(covars, pcs, by = c("FID", "IID"))
# final set of individuals based on genotpe/phenotype overlap
final_sample = covars$IID

# writing covariates out to file
covar_dir = "tteCurated/data/covars/"
write.table(covars, paste0(ukb_dir, covar_dir,"gcta.txt"), quote=F, col.names=F, row.names=F, sep = '\t')
write.table(subset(covars, select = -c(Age)), paste0(ukb_dir,covar_dir,"gcta_noAge.txt"), quote=F, col.names=F, row.names=F, sep = '\t')
covars = subset(covars, select = -c(IID))
colnames(covars)[1] = "sample_id"
write.table(covars, paste0(ukb_dir,covar_dir,"coxmm.txt"), quote=F, col.names=T, row.names=F, sep = '\t')
write.table(subset(covars, select = -c(Age)), paste0(ukb_dir,covar_dir,"coxmm_noAge.txt"), quote=F, col.names=T, row.names=F, sep = '\t')

# now we want to generate the actual censoring event
censoring = censoring[which(censoring$identifier %in% final_sample),]
censoring = subset(censoring, select = c("identifier", "f.54.0.0", "f.191.0.0"))

# last follow up is actually dependent on visit site 
censoring[which(censoring$f.54.0.0 %in% england),"f.54.0.0"] = "England"
censoring[which(censoring$f.54.0.0 %in% scotland),"f.54.0.0"] = "Scotland"
censoring[which(censoring$f.54.0.0 %in% wales),"f.54.0.0"] = "Wales"
censoring[which(censoring$f.54.0.0 == "England"),"f.54.0.0"] = "2022-10-31"
censoring[which(censoring$f.54.0.0 == "Scotland"),"f.54.0.0"] = "2021-07-31"
censoring[which(censoring$f.54.0.0 == "Wales"),"f.54.0.0"] = "2018-02-28"
censoring$f.54.0.0 = as.Date(censoring$f.54.0.0)

# add in death_dt
censoring = merge(censoring, death_dt, by = "identifier", all.x = T)
colnames(censoring)[2:4] = c("last", "lost", "death")
# if there's no lost date set it to today
censoring[which(censoring$lost == ""), "lost"] = as.character(Sys.Date())
censoring$lost = as.Date(censoring$lost)
censoring[which(is.na(censoring$death)), "death"] = as.character(Sys.Date())
censoring$lost = as.Date(censoring$lost)
censoring["last"] = pmin(censoring$death, censoring$last, censoring$lost)
censoring = subset(censoring, select = c(identifier, last))
colnames(censoring) = c("EID", "censored")

# these are the final set of cardio traits analyzed (due to sample size, etc considerations)
cardio_traits = c("Af", "Cad", "DmT2", "Ht", "Hf", "HyperLip", "Mi")

# combine all traits in final file
pheno_dir = "tteCurated/data/pheno/"
all_traits = NULL
all_cardio = NULL
ht_data = NULL
for(trait in cardio_traits){
	print(trait)
	lst.case_control <- get_cases_controls(definitions=dfDefinitions_processed_expanded[which(dfDefinitions_processed_expanded$TRAIT==trait),], lst.harmonized.data$lst.data,dfData.settings, df_reference_date=df_reference_dt_v0)
	data = data.frame(lst.case_control$df.casecontrol)
	colnames(data)[1] = "EID"
	
	# these two conditions should never be true
	if(2 %in% unique(data$Hx)){
		print("Someone was diagnosed before birth")
	}
	if(dim(data[is.na(data$Any),])[1] > 0){
		print("Someone has NA (not excluded) for case-control status, weird")
	}

	# subset to indiduals who were not excluded
	data = data[which(data$Any>0),]

	# subset to individuals with valid reference data (birth date)
	if(dim(data[which(is.na(data$reference_date)),])[1] > 0){
		data = data[-which(is.na(data$reference_date)),]
	}

	# drop anyone who is a case but doesn't have a date of first diagnosis
	if(dim(data[which(data$Any==2 & is.na(data$first_diagnosis_days)),])[1] > 0){	
		data = data[-which(data$Any==2 & is.na(data$first_diagnosis_days)),]
	} 
	data["event"] = 0
	data[which(data$Any==2),"event"] = 1
	data = merge(data, censoring, by = "EID", all.x = T)
	data["censor"] = as.numeric(data$censored - data$reference_date)
	data["stop"] = pmin(data$first_diagnosis_days, data$censor, na.rm=T)
	#if(dim(data[which(data$censor<data$first_diagnosis_days),])[1]>0){
	#	data = data[-which(data$censor<data$first_diagnosis_days),]
	#}
	data[which(data$censor<data$first_diagnosis_days),"event"] = 0
	data["start"] = 0	
	data = subset(data, select = c("EID", "EID", "start", "stop", "event"))
	colnames(data)[1:2] = c("FID","IID")
	
	# convert stop time from days to years (age)
	# not needed for TTE analyses since scale doesn't matter
	# but we use age as covariate so want the appropriate scaling between the two
	data$stop = data$stop/365.25	

	data = data[which(data$FID %in% final_sample),]
        if(trait=="Ht"){
                for(cut in sort(seq(45,75,5),decreasing=T)){
                        new_data = data
                        new_data[which(new_data$stop>cut),"event"] = 0
                        new_data[which(new_data$stop>cut),"stop"] = cut
                        all_traits = rbind(all_traits, c(paste0(trait,"<",cut), dim(new_data[which(new_data$event==1),])[1], dim(new_data)[1]))
        
                        # write out cardio traits to file
                        write.table(subset(new_data, select = c(FID, IID, event)), paste0(ukb_dir, pheno_dir, trait, "_age", cut, "_cc.txt"), quote=F, col.names=F, row.names=F, sep='\t')
                        age = new_data[which(new_data$event==1),]
                        # for age, using quantile normalization which is standard transformation for continous phenotypes
                        age["Age"] = qnorm(rank(age$stop, ties.method = "random")/(length(age$stop)+1))
                        write.table(subset(age, select = c(FID, IID, Age)), paste0(ukb_dir, pheno_dir, trait, "_age", cut, "_age.txt"), quote=F, col.names=F, row.names=F, sep='\t')
                        new_data = subset(new_data, select = -c(IID))
                        colnames(new_data)[1] = "sample_id"
                        write.table(new_data, paste0(ukb_dir, pheno_dir, trait, "_age", cut, "_tte.txt"), quote=F, col.names=T, row.names=F, sep = '\t')
        
                }

        }
	
	all_traits = rbind(all_traits, c(trait, dim(data[which(data$event==1),])[1], dim(data)[1]))
	
	# write out cardio traits to file
	write.table(subset(data, select = c(FID, IID, event)), paste0(ukb_dir, pheno_dir, trait, "_cc.txt"), quote=F, col.names=F, row.names=F, sep='\t')
	age = data[which(data$event==1),]
	# for age, using quantile normalization which is standard transformation for continous phenotypes
	age["Age"] = qnorm(rank(age$stop, ties.method = "random")/(length(age$stop)+1))
	write.table(subset(age, select = c(FID, IID, Age)), paste0(ukb_dir, pheno_dir, trait, "_age.txt"), quote=F, col.names=F, row.names=F, sep='\t')
	data = subset(data, select = -c(IID))
	colnames(data)[1] = "sample_id"
	write.table(data, paste0(ukb_dir, pheno_dir, trait, "_tte.txt"), quote=F, col.names=T, row.names=F, sep = '\t')

	data = subset(data, select = -c(start))
	# columns now sample_id, stop, event
	colnames(data)[2:3] = paste0(colnames(data)[2:3],"_",trait)

	if(trait==cardio_traits[1]){
		all_cardio = data
	} else {
		all_cardio = merge(all_cardio, data, by = "sample_id", all = T)
	}
	
}
write.table(all_cardio, paste0(ukb_dir, pheno_dir, "cardio_traits.txt"), quote=F, col.names=T, row.names=F, sep = '\t')

for(trait1 in cardio_traits){
	#treating these as absorbing states
	# probably makes the most sense clinically
	# leaves Ht, HyperLip, DmT2 -- known risk factors
	if(trait1 %in% c("Af", "Cad", "Hf", "Mi")) next
	col_event1 = which(colnames(all_cardio)==paste0("event_",trait1))
        cardio_temp = all_cardio[which(all_cardio[,col_event1]==1),]
        names = subset(cardio_temp, select = c(sample_id, sample_id))
        #write.table samples
	colnames(names) = c("FID", "IID")
        write.table(names, paste0(ukb_dir, "tteCurated/data/grouping/", trait1, "_samples.txt"),
		col.names=F, row.names=F, quote=F, sep='\t')
        col_time1 = which(colnames(cardio_temp)==paste0("stop_", trait1))
        for(trait2 in cardio_traits){
                if(trait1==trait2) next
                col_names = which(colnames(cardio_temp)==paste0("sample_id"))
                col_event2 = which(colnames(cardio_temp)==paste0("event_",trait2))
                col_time2 = which(colnames(cardio_temp)==paste0("stop_",trait2))
                cardio_temp2 = cardio_temp[,c(col_names, col_time1, col_time2, col_event2)]
                colnames(cardio_temp2) = c("sample_id", "start", "stop", "event")
                cardio_temp2["interval"] = cardio_temp2$stop - cardio_temp2$start
                # treating < 0.0833 or 1 month is rounding error and treating as same time
		if (dim(cardio_temp2[which(cardio_temp2$interval>0.08333),])[1] > 0){
			cardio_temp2 = cardio_temp2[which(cardio_temp2$interval>0.08333),]
                	all_traits = rbind(all_traits, c(paste0(trait1,"->",trait2), dim(cardio_temp2[which(cardio_temp2$event==1),])[1], dim(cardio_temp2)[1]))
			interval = subset(cardio_temp2, select = c(sample_id, start, interval, event))
                	interval$start = 0
               		colnames(interval)[3] = "stop"
                	#write.table interval
                	write.table(interval, paste0(ukb_dir, pheno_dir, trait1, "_", trait2, "_interval.txt"), quote=F, col.names=T, row.names=F, sep = '\t')
			orig = subset(cardio_temp2, select = -c(interval))
                	#write.table orig
                	write.table(orig, paste0(ukb_dir, pheno_dir, trait1, "_", trait2, "_tte.txt"), quote=F, col.names=T, row.names=F, sep = '\t')
			cases = subset(cardio_temp2, select = c(sample_id, sample_id, event))
                	colnames(cases) = c("FID", "IID", "event")
                	#write.table cases
                	write.table(cases, paste0(ukb_dir, pheno_dir, trait1, "_", trait2, "_cc.txt"), quote=F, col.names=F, row.names=F, sep='\t')
			age = cardio_temp2[which(cardio_temp2$event==1),]
			age = subset(age, select = c(sample_id, sample_id, interval))               
			age$interval = qnorm(rank(age$interval, ties.method = "random")/(length(age$interval)+1))
        		colnames(age) = c("FID", "IID", "age")
			#write.table age
			write.table(age, paste0(ukb_dir, pheno_dir, trait1, "_", trait2, "_age.txt"), quote=F, col.names=F, row.names=F, sep='\t')
		}
	}
}

colnames(all_traits) = c('trait', 'cases', 'N')
write.table(all_traits, paste0(ukb_dir, pheno_dir, "all_traits.txt"), quote=F, col.names=T, row.names=F, sep='\t')
