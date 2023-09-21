?boxplot

dat<-data4

t.t <- subset(dat,dat$V2==1)
t.f <- subset(dat,dat$V2==0)

t.true <- subset(t.t, V4==10000)
t.false <- subset(t.f, V4==10000)

t.true.time <- t.t$V3
t.false.time <- t.f$V3

t.a <- cbind(t.true.time, t.false.time)


x<- as.list(as.data.frame(t.a))

names(x)[1] <- "Depleting"
names(x)[2] <- "Non-depleting"

boxplot(x,
        main = "Difference between attractant depletion in square_maze",
        ylab = "number of steps",
        las=2,
        par(cex.axis=0.8),
        horizontal = FALSE)




