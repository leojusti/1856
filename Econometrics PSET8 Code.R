install.packages("plm")
library(plm)

attach(femalelabor)

x <- femalelabor

#x$black <- ifelse(x$type = "black", 1,0 )


summary(lm1 <- lm(ln(wage) ~ status))

summary(lm0 <- lm(log(wage) ~ status + black + hisp + school + ex2 + rur + uwage + ti))

summary(ols <- plm(log(wage) ~ status + black + hisp + school + ex2 + rur + uwage + ti, 
                   model = 'pooling', data = femalelabor[which(!is.na(femalelabor)), ],
                   random.method = "amemiya", na.action=na.exclude))


summary(ols2 <- plm(wage ~ status + black + hisp + school + ex2 + rur + uwage + ti, 
                    model = 'pooling', data = femalelabor))


table(index(femalelabor), useNA = "ifany")

getOption("max.print")
