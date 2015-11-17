# Plot 5

# Reading the Datas
NEI <- readRDS("summarySCC_PM25.rds")
SCC <- readRDS("Source_Classification_Code.rds")


# Transform the datas in order to be able to perform graphics plots
# Find all the texts in SCC$SCC.Level.Two which have the word Vehicles|vehicles
sub <- grep("[Vv]ehicles", SCC$SCC.Level.Two)
# Get all the corresponding indices
indices <- SCC[sub,"SCC"]
# subset the NEI dataset by those indices
NEISub <- NEI[ NEI$SCC %in% indices,]

NEIBal <- subset(NEISub, NEISub$fips == "24510") # Selection of the city of Baltimore

# aggregation : sum by year
mat <- aggregate(Emissions ~ year, data = NEIBal, sum)

# Create PNG file
png(filename = "plot5.png", width = 480, height = 480, units = "px")
g <- ggplot(mat, aes(x = year, y = Emissions))
g <- g + geom_line(col = "blue")
g <- g + ggtitle("Total PM2.5 emissions from motor vehicle sources \n in Baltimore City per year from 1999 to 2008")
g
dev.off()
