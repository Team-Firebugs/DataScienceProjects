# Plot 2

# Reading the Datas
NEI <- readRDS("summarySCC_PM25.rds")
SCC <- readRDS("Source_Classification_Code.rds")


# Transform the datas in order to be able to perform graphics plots
NEIBal <- subset(NEI, NEI$fips == "24510") # Selection of the city of Baltimore
subYear <- tapply(NEIBal$Emissions, NEIBal$year, sum, na.rm=TRUE)

# Create PNG file
png(filename = "plot2.png", width = 480, height = 480, units = "px")
plot(names(subYear), subYear, ylab = "Total PM2.5 emissions", xlab = "Years", main = "Total PM2.5 emissions per year from 1999 to 2008 \n in Baltimore City", type = "l", col = "blue")
dev.off()
