# Plot 6

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

# subset the previous subset NEISub by city "fips" to get only Baltimore City and Los Angeles
NEIFinal <- subset(NEISub, NEISub$fips == "24510" | NEISub$fips == "06037")
# Create a variable "city" in the dataset with the city name (instead of the city code)
NEIFinal$city <- ifelse(NEIFinal$fips == "24510", "Baltimore City", "Los Angeles")

# aggregation : sum by year and city
mat <- aggregate(Emissions ~ year + city, data = NEIFinal, sum)

# Create PNG file
png(filename = "plot6.png", width = 480, height = 480, units = "px")
g <- ggplot(mat, aes(x = year, y = Emissions, col = city))
g <- g + geom_line()
g <- g + ggtitle("Comparison of emissions from motor vehicle sources \n from 1999 to 2008 in Baltimore City and Los Angeles")
g
dev.off()
