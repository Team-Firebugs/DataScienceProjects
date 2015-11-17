library(shiny)
library(ggplot2)

# First, we load the dataset outside of the shinyServer since we only have to
# load it once.
data(mtcars)
datac <- mtcars
datac$cyl <- factor(datac$cyl)
datac$vs <- factor(datac$vs)
datac$am <- factor(datac$am)
datac$gear <- factor(datac$gear)
datac$carb <- factor(datac$carb)

# Shiny Server
shinyServer(function(input, output) {

    # We define the reactive expressions. They are meant to be called whenever
    # the inupts change.
    # So we set x and y to be equal to the appropriate vector depending on the
    # choice that has been taken with the radioButton.
    x <- reactive({ switch(input$xAxis, "mpg" = datac$mpg,
                           "disp" = datac$disp, 
                           "hp" = datac$hp, 
                           "wt" = datac$wt,
                           datac$wt) }) 
    y <- reactive({ switch(input$yAxis, "mpg" = datac$mpg,
                           "disp" = datac$disp,
                           "hp" = datac$hp,
                           "wt" = datac$wt,
                           datac$mpg) })
    
    # Then we set n to be equal to itself, but as a reactive variable.
    n <- reactive ({ input$n })
    
    # Finally, we set the fit expression as a reactive expression.
    fit <- reactive ({ lm(y()[1:n()] ~ x()[1:n()], data=data.frame(x=x()[1:n()],y=y()[1:n()])) })
    
    
    # Here we generate the plot. Note that we call here the reactive variables
    # and reactive expressions inside the plots commands, so that the values
    # are updated as the inputs change.
    output$plot <- renderPlot({
        ggplot(data.frame(x =x()[1:n()], y = y()[1:n()]), aes(x,y))+
            geom_point(color = "steelblue", size = 6, alpha = 3/4)+
            labs(title = paste(input$yAxis, "vs", input$xAxis, sep=" "))+
            labs(x = input$xAxis) + labs(y = input$yAxis)+
            scale_x_continuous(limits = c( min(x()), max(x() )) )+
            scale_y_continuous(limits = c( min(y()), max(y() )) )+
            geom_abline(intercept = coef(fit())[1],
                        slope = coef(fit())[2], colour = "red")+
            theme_bw(base_family = "Times")
        
    })
    
    # We print here the summary of the fit model
    output$summary <- renderPrint({
        summary(fit())
    })
    
    # Here we print the data.frame for the number of observations selected with
    # the slider in the lef side panel.
    output$table <- renderTable({
        data.frame(datac[1:n(),])
    })
})