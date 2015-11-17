library(shiny)

shinyUI(
    # This is the main title of the navigation bar, which contains the 2 global
    # tabs
    navbarPage("mtcars DATASET ANALYSIS",    
               # This is the title of the first global tab
               tabPanel("Application",
                        
                        # Here we set the left side panel
                        sidebarPanel(
                            # We generate radioButtons for choosing the x axis.
                            radioButtons("xAxis", "Choose a variable to put on the x-axis:",
                                         list("MPG (Miles per Gallon" = "mpg",
                                              "Displacement (cu.in.)" = "disp",
                                              "Gross Horse Power" = "hp",
                                              "Weight (lb/1000)" = "wt")),
                            # And we generate others radioButtons for choosing the y axis.
                            radioButtons("yAxis", "Choose a variable to put on the y-axis:",
                                         list("MPG (Miles per Gallon" = "mpg",
                                              "Displacement (cu.in.)" = "disp",
                                              "Gross Horse Power" = "hp",
                                              "Weight (lb/1000)" = "wt")),
                            # br() is meant to create a vertical space
                            br(),
                            # We create the slider for the number of observations
                            sliderInput("n", 
                                        "Number of observations:", 
                                        value = 32,
                                        min = 2, 
                                        max = 32)
                        ),
                        # Then we generate the main panel, in which we display 3
                        # tabs, one for the plot, one for the summary, and one for
                        # the dataset as a table.
                        mainPanel(
                            tabsetPanel(
                                tabPanel("Plot", plotOutput("plot")), 
                                tabPanel("Fit Summary", br(), verbatimTextOutput("summary")), 
                                tabPanel("Dataset", br(), tableOutput("table"))
                            )
                        )
               ),
               # Finally, we set the second global tab, inserting only the
               # documentation as the main page which is contained in the 
               # "about.md" file.
               tabPanel("About",
                        mainPanel(
                            includeMarkdown("about.md")
                        ))
    )
)