# Plot 3

# Reading the Datas
NEI <- readRDS("summarySCC_PM25.rds")
SCC <- readRDS("Source_Classification_Code.rds")


# Transform the datas in order to be able to perform graphics plots
NEIBal <- subset(NEI, NEI$fips == "24510") # Selection of the city of Baltimore

# aggregation : sum by year and type
mat <- aggregate(Emissions ~ type + year, data = NEIBal, sum)

# Create PNG file
png(filename = "plot3.png", width = 680, height = 480, units = "px")
library(ggplot2)
g <- ggplot(mat, aes(x = year, y = Emissions))
g <- g + geom_line(col="blue")
g <- g + facet_grid(. ~ type)
g <- g + ggtitle("Emissions from 1999 to 2008 in Baltimore City \n for each type")
g
dev.off()
