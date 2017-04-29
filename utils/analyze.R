dt <- .00005

print('Loading...')
w <- as.matrix(read.table('wOut.dat', header=F))
r <- as.matrix(read.table('r.dat', header=F))
z <- diag(w%*%r)
time = seq(0, dt*length(z)*100, length.out = length(z))

tgt <- sin(8*pi*time) + 2*cos(4*pi*time)

plot(time, z, t = 'l', col = 'red', xlab = 'time (s)', ylab = 'target')
lines(time, tgt, col = 'blue', lwd = 3)

#plot(apply(r, 1, max), t = 'l', col = 'black', lwd = 2, xlim = c(0,nrow(r)), ylim= c(0, max(as.vector(r))))
#lines(apply(r, 1, min), col = 'black', lwd = 2)
#lines(apply(r, 1, median), col = 'grey')

#lines(r[,which.max(apply(r, 2, mean))], col = 'blue', lwd = 3)
#lines(r[,which.min(apply(r, 2, mean))], col = 'red', lwd = 3)

# Quantile plot

qs <- t(apply(r, 2, quantile))

matplot(qs, t = 'l', main = 'Quantiles', xlab = 'Time (5 ms)', ylab = 'Firing rate (Hz)',
        col = rev(rainbow(15))[1:ncol(qs)])

# Find how many neurons aren't firing

threshold <- 1/(100*nrow(r))

unfired <- sort(apply(r, 1, function(x) sum(x < threshold)/length(x)))

plot(unfired, xlab = 'Neuron', ylab = 'Fraction of time firing below threshold', t = 'l')
