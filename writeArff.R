# Output to ARFF file format for use with Weka

waikato.write.arff <- function (x, file = "data", ncolumns = if (is.charactor(x)) 1 else 5, append = FALSE, class_col = NULL){
  if (append == TRUE){
    stop("Append not yet supported")
  }

  if(is.character(file)) {
      file <- file(file,"w")
      on.exit(close(file))
  }
  
  if (!inherits(file,"connection"))
      stop("argument `file' must be a character string or connection")

  if (!isOpen(file)) {
      open(file,"r")
      on.exit(close(file))
  }

  # Write header
  writeLines("% Output from R",file)
  writeLines("@relation ROutput",file)

  colNames <- dimnames(x)[[2]]
  for (colName in colNames){
    if (is.factor(x[1,colName])){
      values <- unique(x[colName])[,1]
      writeLines(paste("@attribute '", colName, "' {", paste(values,collapse=","), "}", sep=""),file)
    } else {
      colType <- "REAL"
      writeLines(paste("@attribute '", colName, "' ", colType, sep=""),file)
    }
  }

  writeLines("@data", file)
  # Write data
  rows <- c()
  for (i in 1:nrow(x)) {
    # FIXME This messy code is to ensure that the factor types stay as factors
    row <- ""
    for (c in x[i,]){
      if (row == "")
        row <- c
      else
        row <- paste(row,c,sep=",")
    }
    rows <- c(rows, row)
    if (length(rows) > 50){
      writeLines(rows, file)
      rows <- c()
    }
  }

  if (length(rows) > 0){
    writeLines(rows, file)
    rows <- c()
  }
  
}