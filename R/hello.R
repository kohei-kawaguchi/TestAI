#' Hello World Function
#'
#' This function prints a greeting message.
#'
#' @param name Character string. Name to greet. Default is "World".
#' @return A character string with the greeting message.
#' @export
#' @examples
#' hello_world()
#' hello_world("Alice")
hello_world <- function(name = "World") {
  message <- paste0("Hello, ", name, "!")
  print(message)
  return(invisible(message))
}
