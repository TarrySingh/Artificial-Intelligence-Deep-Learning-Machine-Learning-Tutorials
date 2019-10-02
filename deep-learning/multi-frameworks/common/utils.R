# Create an array of fake data to run inference on
give_fake_data <- function(batches, col_major = FALSE){
  set.seed(0)
  if (col_major) {
    shape <- c(224, 224, 3, batches)
  } else {
    shape <- c(batches, 224, 224, 3)
  }
  dat <- array(runif(batches*224*224*3), dim = shape)
  return(dat)
}

# Return features from classifier (OLD)
predict_fn <- function(classifier, data, batchsize){
    out <- array(0, dim = c(dim(data)[1], params$RESNET_FEATURES))
    idx <- 0:(dim(data)[1] %/% batchsize - 1)
    for (i in idx){
        dta <- data[(i*batchsize + 1):((i+1)*batchsize),,,]
        out[(i*batchsize + 1):((i+1)*batchsize), ] <- predict_on_batch(classifier, dta)
    }
    return(out)
}


# Get GPU name
get_gpu_name <- function(){
    tryCatch(
        {
            out_list <- system("nvidia-smi --query-gpu=gpu_name --format=csv", intern = TRUE)
            out_list <- out_list[out_list != "name"]
            return(out_list)
        },
        error = function(e)
        {
            print(e)
        }
        )
}

# Get CUDA version
get_cuda_version <- function(){
    tryCatch(
        {
            out <- system("cat /usr/local/cuda/version.txt", intern = TRUE)
            return(out)
        },
        error = function(e)
        {
            print(e)
        }
        )
}

# Get CuDNN version
get_cudnn_version <- function(){
    tryCatch(
        {
            out <- system("cat /usr/include/cudnn.h | grep CUDNN_MAJOR", intern = TRUE)[1]
            indx <- regexpr("(\\d+)", out)
            major <- regmatches(out, indx)
            
            out <- system("cat /usr/include/cudnn.h | grep CUDNN_MINOR", intern = TRUE)[1]
            indx <- regexpr("(\\d+)", out)
            minor <- regmatches(out, indx)
            
            out <- system("cat /usr/include/cudnn.h | grep CUDNN_PATCHLEVEL", intern = TRUE)[1]
            indx <- regexpr("(\\d+)", out)
            patch <- regmatches(out, indx)
            
            version <- paste(major, minor, patch, sep = ".")
            return(paste0("CuDNN Version ", version))
        },
        error = function(e)
        {
            print(e)
        }
        )
}



# Function to download the cifar data, if not already downloaded
maybe_download_cifar <- function(col_major = TRUE, src = 'https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/cifar-10-binary.tar.gz '){
  
  tryCatch(
    {
      data <- suppressWarnings(process_cifar_bin(col_major))
      return(data)
    },
    error = function(e)
    {
      print(paste0('Data does not exist. Downloading ', src))
      download.file(src, destfile="tmp.tar.gz")
      print('Extracting files ...')
      untar("tmp.tar.gz")
      file.remove('tmp.tar.gz')
      return(process_cifar_bin(col_major))
    }
  )
}


# A function to process CIFAR10 dataset in binary format
process_cifar_bin <- function(col_major) {
  
  data_dir <- "cifar-10-batches-bin"
  
  train <- lapply(file.path(data_dir, paste0("data_batch_", 1:5, ".bin")), read_file)
  train <- do.call(c, train)
  
  x_train <- unlist(lapply(train, function(x) x$image))
  if (col_major) {
    perm <- c(2, 1, 3, 4)
  } else {
    perm <- c(4, 3, 2, 1)
  }
  
  x_train <- aperm(array(x_train, c(32, 32, 3, 50000)), perm = perm)
  x_train <- x_train / 255
  y_train <- unlist(lapply(train, function(x) x$label))
  
  test <- read_file(file.path(data_dir, "test_batch.bin"))
  x_test <- unlist(lapply(test, function(x) x$image))
  x_test <- aperm(array(x_test, c(32, 32, 3, 10000)), perm = perm)
  x_test <- x_test / 255
  y_test <- unlist(lapply(test, function(x) x$label))
  
  list(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)
}

                          
                          
# A function to load CIFAR10 dataset
cifar_for_library <- function(one_hot = FALSE, col_major = TRUE) {
  
  cifar <- maybe_download_cifar(col_major)
  
  x_train <- cifar$x_train
  y_train <- cifar$y_train
  x_test <- cifar$x_test
  y_test <- cifar$y_test
  
  if(one_hot){
    Y = data.frame(label = factor(y_train))
    y_train = with(Y, model.matrix(~label+0))
    Y = data.frame(label = factor(y_test))
    y_test = with(Y, model.matrix(~label+0))
  }
  
  list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
  
}                          

# Load hyper-parameters for different scenarios:
# cnn, lstm, or inference
load_params <- function(params_for){
    
    require(rjson)
    params <- fromJSON(file = "./common/params.json")

    if (params_for == "cnn"){
        return(params$params_cnn)
    } else if (params_for == "lstm"){
        return(params$params_lstm)
    } else if (params_for == "inference"){
        return(params$params_inf)
    } else {
        stop("params_for should be set to one of the following: cnn, lstm or inference.")
    }
}


# Function to download the mxnet resnet50 model, if not already downloaded
maybe_download_resnet50 <- function() {
  src <- 'http://data.mxnet.io/models/imagenet/'
  tryCatch(
    {
      model <- suppressWarnings(mx.model.load(prefix = "resnet-50", iteration = 0))
      return(model)
    },
    error = function(e)
    {
      print(paste0('Model does not exist. Downloading ', src))
      download.file(file.path(src, 'resnet/50-layers/resnet-50-symbol.json'), destfile="resnet-50-symbol.json")
      download.file(file.path(src, 'resnet/50-layers/resnet-50-0000.params'), destfile="resnet-50-0000.params")
      return(mx.model.load(prefix = "resnet-50", iteration = 0))
    }
  )
}

load_resnet50 <- function() maybe_download_resnet50()

read_image <- function(i, to_read) {
  label <- readBin(to_read, integer(), n = 1, size = 1)
  image <- as.integer(readBin(to_read, raw(), size = 1, n = 32*32*3))
  list(label = label, image = image)
}


read_file <- function(f) {
  to_read <- file(f, "rb")
  examples <- lapply(1:10000, read_image, to_read)
  close(to_read)
  examples
}

# Plot a CIFAR10 image
plot_image <- function(img) {
  library(grid)
  img_dim <- dim(img)
  if (img_dim[1] < img_dim[3]) {
    r <- img[1,,]
    g <- img[2,,]
    b <- img[3,,]
  } else {
    r <- img[,,1]
    g <- img[,,2]
    b <- img[,,3]
  }
  img.col.mat <- rgb(r, g, b, maxColorValue = 1)
  dim(img.col.mat) <- dim(r)
  grid.raster(img.col.mat, interpolate = FALSE)
  rm(img.col.mat)
}


maybe_download_imdb <- function(src = 'https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/imdb.Rds'){
  
  tryCatch(
    {
      data <- suppressWarnings(readRDS("imdb.Rds"))
      return(data)
    },
    error = function(e)
    {
      print(paste0('Data does not exist. Downloading ', src))
      download.file(src, destfile="imdb.Rds")
      return(readRDS("imdb.Rds"))
    }
  )
}


imdb_for_library <- function() maybe_download_imdb()
    

