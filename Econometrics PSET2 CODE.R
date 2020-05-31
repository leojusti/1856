library(quantreg)
data(barro)

barro<-subset(barro,select=-c(X1))
deciles<-seq(0.1,0.9,0.1)


rqfit <- rq(y.net	~ lgdp2	+mse2	+fse2	+fhe2	+mhe2	+lexp2	+lintr2	+gedy2
            +Iy2	+gcony2	+lblakp2+	pol2	+ttrad2, tau = deciles, data = barro)

rqfit2 <- rq(y.net	~. , tau = deciles, data = barro)

summary(rqfit)
summary(rqfit2)
par(mar=c(1,1,1,1))
plot(summary(rqfit))
plot(summary(rqfit2))

rqfit1 <- summary(rqfit2, se="boot", bsmethod="mcmb")
rqfit1

x <- as.data.frame(rqfit1)

latex.summary.rqs(rqfit1, tranpose=FALSE, caption="Quantile Regression Table", digits=2)
