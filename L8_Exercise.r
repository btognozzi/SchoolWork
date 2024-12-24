# this is how you save a plot to a file
library(ggplot2)

# comment this out to see the plot in the notebook 
png(filename="myplot.png")

# your plot here..
qplot(carat, price, data = diamonds)

# comment this out to see the plot in the notebook
dev.off()

# Read your data here; 

df <- read.csv("MO_diabetes.csv", stringsAsFactors = FALSE)

# Clean it up, reformat it, filter it, sort it, group it, etc.

df$County <- tolower(gsub("County", "", df$County))
df$ex_op <- as.numeric(df$ex_op)
df$pop <- as.numeric(gsub(",", "", df$pop))

df_sub <- subset(df, select = c("County", "pop", "dia_per", "ob_per", "hlth_ins_per", "smoke", "chol", "hbp",
                               "phys_inact", "hlth_food", "sev_hou_cost", "dip_per", "pov_per"))

df_sub$dia_pop <- df_sub$pop * (df_sub$dia_per / 100)
df_sub$ob_pop <- df_sub$pop * (df_sub$ob_per / 100)
df_sub$hlth_ins_pop <- df_sub$pop * (df_sub$hlth_ins_per / 100)
df_sub$smoke_pop <- df_sub$pop * (df_sub$smoke / 100)
df_sub$chol_pop <- df_sub$pop * (df_sub$chol / 100)
df_sub$hbp_pop <- df_sub$pop * (df_sub$hbp / 100)
df_sub$inact_pop <- df_sub$pop * (df_sub$phys_inact / 100)
df_sub$food_pop <- df_sub$pop * (df_sub$hlth_food / 100)
df_sub$house_pop <- df_sub$pop * (df_sub$sev_hou_cost / 100)
df_sub$dip_pop <- df_sub$pop * (df_sub$dip_per / 100)
df_sub$pov_pop <- df_sub$pop * (df_sub$pov_per / 100)
                   

# Create your visualizations and save them as png files, then prepare your final pdf document elsewhere

# <- ALL YOUR CODE BELOW THIS POINT ->



library(reshape2)
cor_mat <- melt(cor(df_sub[,3:ncol(df_sub)]))

library(tidyverse)
library(RColorBrewer)

heatmap <- cor_mat %>% 
ggplot(aes(Var1, Var2, fill = value)) + geom_tile() + geom_text(label = round(cor_mat$value, 2), size = 2.5) +
scale_fill_gradient(low = "white", high = "red") +
xlab("") +
ylab("") +
ggtitle("Correlation Heatmap") +
theme_minimal() +
theme(axis.ticks = element_blank(), 
      axis.text.x = element_text(size = 10, angle = 90, hjust = 0, color = "black"),
      axis.text.y = element_text(size = 10, color = "black"))

heatmap

# Calculating P-values for the high correlation variables to diabetes population
p1 <- t.test(df_sub$dia_pop, df_sub$food_pop)$p.value
p2 <- t.test(df_sub$dia_pop, df_sub$ob_pop)$p.value
p3 <- t.test(df_sub$dia_pop, df_sub$inact_pop)$p.value
p4 <- t.test(df_sub$dia_pop, df_sub$house_pop)$p.value
p5 <- t.test(df_sub$dia_pop, df_sub$hlth_ins_pop)$p.value
p6 <- t.test(df_sub$dia_pop, df_sub$smoke_pop)$p.value
p7 <- t.test(df_sub$dia_pop, df_sub$chol_pop)$p.value
p8 <- t.test(df_sub$dia_pop, df_sub$hbp_pop)$p.value
p9 <- t.test(df_sub$dia_pop, df_sub$dip_pop)$p.value
p10 <- t.test(df_sub$dia_pop, df_sub$pov_pop)$p.value

# Output the P-values for each high correlation variable.  Only select statistically significant values (between 0 and 0.05)
p1
p2
p3
p4
p5
p6
p7
p8
p9
p10

# Variables using for analysis: ob_pop, inact_pop, smoke_pop, chol_pop, hbp_pop, dip_pop, pov_pop
f_df <- subset(df_sub, select = c("County", "pop", "dia_pop",
                                 "ob_pop", "inact_pop", "smoke_pop",
                                 "chol_pop", "hbp_pop", "dip_pop", "pov_pop"))

summary(f_df)

f_df <- f_df %>% filter(dip_pop > 0 & pov_pop > 0)

log_df <- log10(f_df[,3:ncol(f_df)])
head(log_df)

final_df <- cbind(f_df$County, f_df$pop, log_df, stringsAsFactors = FALSE)
head(final_df)

names(final_df)[names(final_df) == 'f_df$County'] <- 'county'
names(final_df)[names(final_df) == 'f_df$pop'] <- 'pop'

head(final_df)

summary(final_df)

cor(final_df[,3:ncol(final_df)])

set.seed(20)
sample <- sample(c(TRUE, FALSE), nrow(final_df), replace=TRUE, prob=c(0.7,0.3))
train  <- final_df[sample, ]
test   <- final_df[!sample, ]

head(train)
head(test)

m1 <- lm(dia_pop ~ ob_pop + inact_pop, data = train)
m2 <- lm(dia_pop ~ smoke_pop + chol_pop + hbp_pop, data = train)
m3 <- lm(dia_pop ~ dip_pop + pov_pop, data = train)

summary(m1)
summary(m2)
summary(m3)

test$y_pred_m1 <- predict(m1, test)
test$y_pred_m2 <- predict(m2, test)
test$y_pred_m3 <- predict(m3, test)

head(test)

test$e_m1 <- abs(test$dia_pop - test$y_pred_m1)
test$e_m2 <- abs(test$dia_pop - test$y_pred_m2)
test$e_m3 <- abs(test$dia_pop - test$y_pred_m3)
head(test)

library(gridExtra)

plot1 <- ggplot(test, aes(dia_pop, y_pred_m1, color = e_m1)) + geom_point() + geom_smooth(method = "lm", color = "blue", se = FALSE) +
         scale_color_gradient(low = "white", high = "red") + 
         theme_minimal() +
         ggtitle("Diabetes Prediction based on Traditional Risk Factors")
plot2 <- ggplot(test, aes(dia_pop, y_pred_m2, color = e_m2)) + geom_point() + geom_smooth(method = "lm", color = "blue", se = FALSE) +
         scale_color_gradient(low = "white", high = "red") + 
         theme_minimal() +
         ggtitle("Diabetes Predicition based on Strongly Correlated Risk Facotrs")
plot3 <- ggplot(test, aes(dia_pop, y_pred_m3, color = e_m3)) + geom_point() + geom_smooth(method = "lm", color = "blue", se = FALSE) +
         scale_color_gradient(low = "white", high = "red") + 
         theme_minimal() +
         ggtitle("Diabetes Predicition based on Socioeconomic Factors")

grid.arrange(plot1, plot2, plot3)

names(df_sub)[names(df_sub) == 'County'] <- 'county'

head(df_sub)

mo <- map_data('county', 'missouri')
mo$region = mo$subregion

dia_df <- subset(df_sub, select = c("county", "dia_per"))

head(dia_df)

dia_df$county <- gsub("\\.", "", dia_df$county)

trim <- function (x) gsub("^\\s+|\\s+$", "", x)

dia_df$county <- trim(dia_df$county)
dia_df$dia_per <- trim(dia_df$dia_per)

dia_df$dia_per <- as.numeric(dia_df$dia_per)

dia_df[32, 1] = "de kalb"

dia_mo_map <- ggplot() +

geom_map(data = mo, map = mo, aes(map_id = region), fill = NA, color = "black") + 

geom_map(data = dia_df, map = mo, aes(map_id = county, fill = dia_per)) + 

scale_fill_distiller(type = "seq", palette = "YlOrRd", direction = 1, name = "") + 

expand_limits(x = mo$long, y = mo$lat) +
coord_map("polyconic") + 
theme_void()

dia_mo_map

df_m1 <- test %>%
  select(dia_pop, y_pred_m1, e_m1) %>%
  rename(Predicted = y_pred_m1, Error = e_m1) %>%
  mutate(Model = "m1")

df_m2 <- test %>%
  select(dia_pop, y_pred_m2, e_m2) %>%
  rename(Predicted = y_pred_m2, Error = e_m2) %>%
  mutate(Model = "m2")

df_m3 <- test %>%
  select(dia_pop, y_pred_m3, e_m3) %>%
  rename(Predicted = y_pred_m3, Error = e_m3) %>%
  mutate(Model = "m3")

# Combine data frames
combined_df <- bind_rows(df_m1, df_m2, df_m3)

coefficients_df <- data.frame(
  Model = c("m1", "m2", "m3"),
  Intercept = c(coef(m1)[1], coef(m2)[1], coef(m3)[1]),
  Slope = c(coef(m1)[2], coef(m2)[2], coef(m3)[2])
)

coefficients_df$Model <- as.character(coefficients_df$Model)

x_limits <- range(combined_df$dia_pop)
y_limits <- range(combined_df$Predicted)

error_plot <- ggplot(combined_df, aes(dia_pop, Predicted, color = Error)) +
  geom_point() +
  stat_smooth(method = "lm", geom='line', alpha=0.5, se=FALSE, color = "blue") +
  scale_color_gradient(low = "gray", high = "red") +
  theme_minimal() +
  xlim(x_limits) +
  ylim(y_limits) +
  xlab("Actual") +
  ylab("Predicted") + 
  facet_wrap(~Model, scales = "free", labeller = as_labeller(c(m1 = "Model 1", m2 = "Model 2", m3 = "Model 3"))) +
  theme(strip.text = element_text(hjust = 0.5),
        axis.title.x = element_text(hjust = 0.5),
        axis.title.y = element_text(hjust = 0.5),
        plot.title = element_text(hjust = 0.5))

error_plot

cor_mat2 <- melt(cor(final_df[,3:ncol(final_df)]))

heatmap2 <- cor_mat2 %>% 
ggplot(aes(Var1, Var2, fill = value)) + geom_tile() + 
scale_fill_gradient(low = "yellow", high = "red", limit = c(0,1)) +
xlab("") +
ylab("") +
theme_minimal() +
theme(panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(),
      axis.ticks = element_blank(), 
      axis.title.x = element_text(hjust = 0.5),
      axis.text.x = element_text(size = 10, angle = 90, hjust = 0, color = "black"),
      axis.text.y = element_text(size = 10, color = "black"))

heatmap2

# Exporting mo_map plot
png(filename="dia_mo_map.png")

# your plot here..
dia_mo_map

dev.off()

# Exporting mo_map plot
png(filename="error_plot.png")

# your plot here..
error_plot

dev.off()

# Exporting mo_map plot
png(filename="correlation.png")

# your plot here..
heatmap2

dev.off()

mse1 <- mean((test$dia_pop - test$y_pred_m1)^2)
mse2 <- mean((test$dia_pop - test$y_pred_m2)^2)
mse3 <- mean((test$dia_pop - test$y_pred_m3)^2)

rmse1 <- sqrt(mse1)
rmse2 <- sqrt(mse2)
rmse3 <- sqrt(mse3)

mse1
mse2
mse3
rmse1
rmse2
rmse3
