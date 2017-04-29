dt <- 5e-5
step <- 100

files <- list.files(path = '~/cpp/cppTranslated/', pattern = 'r_N_[0-9]+')

r.avg <- r.sd <- unf <-  matrix(0, 5, 5)
rownames(r.avg) <- rownames(r.sd) <- rownames(unf) <- c('0.01', '0.05', '0.1', '0.5', '1')
colnames(r.avg) <- colnames(r.sd) <- colnames(unf) <- c('1000', '2000', '3000', '4000', '7000')

r.eq <- c()

#plot(0, 0, t = 'n', xlim = c(0, 1), ylim = c(0, 1),
#     xlab = 'Neuron', ylab = 'Fraction of time firing below threshold')

for (i in 1:length(files))
{
  file <- files[i]
  
  params <- stringr::str_split(file, '_')[[1]]
  N <- params[3]
  G <- params[5]

  print(paste0('Reading file... ', i, ' out of ', length(files)))
  r <- as.matrix(read.table(file, header = F))
  
  r.eq <- rbind(r.eq, apply(r, 2, mean))

  #r.avg[G, N] <- r.avg[G, N] + mean(as.vector(r))
  #r.sd[G, N] <- r.sd[G, N] + sd(as.vector(r))

  threshold <- 1/(step*ncol(r))
  #unfired <- sort(apply(r, 1, function(x) sum(x < threshold)/length(x)))
  #lines(seq(0, 1, length.out = nrow(r)), unfired, col = i)

  #unf[G, N] <- unf[G,N] + sum(unfired > .95)/length(unfired)
}

r.avg <- r.avg/9
r.sd <- r.sd/9
unf <- unf/9

#legend('topleft', legend = c(1000, 2000, 3000, 4000, 7000), bty = 'n', lty = 1, col = 1:length(files))
