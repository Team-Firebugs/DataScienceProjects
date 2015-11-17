# Plot 4

# Reading the Datas
NEI <- readRDS("summarySCC_PM25.rds")
SCC <- readRDS("Source_Classification_Code.rds")


# Transform the datas in order to be able to perform graphics plots
# Find all the short names in SCC$Short.name which have the word Coal|coal
sub1 <- grep("[Cc]oal", SCC$Short.Name)
# Get all the corresponding indices
indices <- SCC[sub1,"SCC"]
# subset the NEI dataset by those indices
NEISub <- NEI[ NEI$SCC %in% indices,]

# aggregation : sum by year
mat <- aggregate(Emissions ~ year, data = NEISub, sum)

# Create PNG file
png(filename = "plot4.png", width = 480, height = 480, units = "px")
g <- ggplot(mat, aes(x = year, y = Emissions))
g <- g + geom_line(col = "blue")
g <- g + ggtitle("Total PM2.5 emissions from coal combustion-related sources \n per year from 1999 to 2008")
g
dev.off()
