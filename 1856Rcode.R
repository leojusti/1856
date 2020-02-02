
library(ggplot2)


attach(yield_data)
attach(spot_data)


##yield
x <- ggplot(spot_data, aes(x=Date, y=Yield, color = Curve, group = Curve)) +
  geom_line() +
  labs(x = "Maturity Date", y = "Yield", 
       title = "All Yield Curves") + 
  theme_minimal() +
  ylim(0,0.03)

ggsave(x, filename = "sp.png", height = 4 , width = 7)


##spot
y <- ggplot(yield_data, aes(x=Year, y=Yield, color = Curve, group = Curve)) +
  geom_line() +
  labs(x = "Maturity Date", y = "Yield", 
       title = "All Spot Curves") + 
  theme_minimal() +
  ylim(0,0.013)

ggsave(y, filename = "yp.png", height = 4 , width = 7)

##forward
z <- ggplot(for_data, aes(x=Year, y=Yield, color = Curve, group = Curve)) +
  geom_line() +
  labs(x = "Year", y = "f(t,T)", 
       title = "All Forward Curves") + 
  theme_minimal() +
  ylim(0,0.1)

ggsave(z, filename = "fp.png", height = 4 , width = 7)

a <- c(-0.020363999,
-0.010641125,
0.005043367,
0.002554224,
-0.00128653,
0.012179678,
-0.007642312,
-0.004204183,
-0.017439267)

b <- c(-0.019433383,
       -0.012285311,
       0.00374869,
       0.008451241,
       -0.000953025,
       0.01762125,
       -0.012100305,
       -0.000934922,
       -0.015013913)

c <- c(-0.021510335,
       -0.014614878,
       0.005372837,
       0.000990515,
       0.016833914,
       0.005161952,
       0.006468205,
       -0.008011334,
       -0.014316938)


d <- c(-0.013577399,
       -0.008644633,
       0.004486787,
       -0.001784919,
       0.009440279,
       0.000596073,
       0.009880792,
       -0.006554776,
       -0.008889109)

e <- c(-0.042529464,
       -0.031374641,
       0.018718906,
       -0.001552478,
       0.027897132,
       -0.007774839,
       0.004710099,
       -0.006064392,
       -0.015365034)

M <- cbind(a,b,c,d,e)

cov(M)
E <- cov(M)
E
u <- eigen(E)


f <- c(-0.016316977,
       -0.018189722,
       -0.000827361,
       0.029274749,
       0.000197471,
       0.036398106,
       -0.027431847,
       0.010266758,
       -0.006907118)

g <- c(-0.022018914,
       -0.015663145,
       0.005508253,
       0.00065582,
       0.021005855,
       0.00369475,
       0.009607143,
       -0.008928478,
       -0.013813644)

h <- c(-0.013467714,
       -0.008746043,
       0.004574751,
       -0.002158595,
       0.010507516,
       -0.00024454,
       0.011463822,
       -0.006919133,
       -0.008523516)

j <- c(-0.027564473,
       -0.031994568,
       0.017749696,
       -0.007840325,
       0.029267088,
       0.001248333,
       -0.001722909,
       -0.005793033,
       -0.016822847)

N <- cbind(f,g,h,j)

cov(N)
j <- cov(N)

E1 <- eigen(cov(N))
E1

