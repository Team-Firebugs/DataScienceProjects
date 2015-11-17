# Plot 1

# Reading the Datas
NEI <- readRDS("summarySCC_PM25.rds")
SCC <- readRDS("Source_Classification_Code.rds")


# Transform the datas in order to be able to perform graphics plots
subYear <- tapply(NEI$Emissions, NEI$year, sum, na.rm=TRUE)

# Create PNG file
png(filename = "plot1.png", width = 480, height = 480, units = "px")
plot(names(subYear), subYear, ylab = "Total PM2.5 emissions", xlab = "Years", main = "Total PM2.5 emissions per year from 1999 to 2008 \n in the United States", type = "l", col = "blue")
dev.off()
