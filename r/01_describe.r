library(tidyverse)  # 1.3.1
library(ggcorrplot) # 0.1.3
library(OpenML)

#====================================================================
# Import and inspect
#====================================================================

raw <- OpenML::getOMLDataSet(43093)$data
summary(raw)

# Proportion resellings
mean(duplicated(raw$PARCELNO))  # 0.01119724

#====================================================================
# Map
#====================================================================

# Simple price map with logarithmic price per sqft color scale
raw %>%
  mutate(price_per_sqft = SALE_PRC / TOT_LVG_AREA) %>%
ggplot(aes(LONGITUDE, LATITUDE, z = price_per_sqft)) +
  coord_quickmap() +
  stat_summary_2d(bins = 50) +
  labs(fill = "Mean price per sqft",
       x = element_blank(),
       y = element_blank()) +
  scale_fill_viridis_c(option = "inferno", begin = 0.2, trans = "log10") +
  theme_void() +
  theme(axis.ticks = element_blank(),
        axis.text = element_blank(),
        strip.text.x = element_text(size = 12),
        legend.position = c(0.7, 0.1),
        legend.key.height = unit(3, units = "mm"),
        legend.background = element_rect(fill = "transparent",
                                         linetype = "blank"),
        plot.margin=grid::unit(c(0, 0, 0, 0), "mm"))

#====================================================================
# Prepare data
#====================================================================

prep <- raw %>%
  mutate(
    log_price = log(SALE_PRC),
    log_living = log(TOT_LVG_AREA),
    log_land = log(LND_SQFOOT),
    log_special = log1p(SPEC_FEAT_VAL),
    s_dist_highway = sqrt(HWY_DIST / 1000),
    s_dist_rail = sqrt(RAIL_DIST / 1000),
    dist_ocean = OCEAN_DIST / 1000,
    log_dist_water = log1p(WATER_DIST / 1000),
    s_dist_center = sqrt(CNTR_DIST / 1000),
    s_dist_sub = sqrt(SUBCNTR_DI / 1000)
  )

#====================================================================
# Variable groups
#====================================================================

x_continuous <- c("log_living", "log_land", "log_special", "age",
                  "s_dist_highway", "s_dist_rail", "dist_ocean",
                  "log_dist_water", "s_dist_center", "s_dist_sub")
x_coord <- c("LATITUDE", "LONGITUDE")
x_discrete <- c("month_sold", "structure_quality")
x_vars <- c(x_continuous, x_coord, x_discrete)
y_var <- "log_price"

#====================================================================
# Univariate description
#====================================================================

# Histograms
x <- c(y_var, x_continuous)
prep %>%
  select_at(x) %>%
  pivot_longer(everything()) %>%
  mutate(name = factor(name, levels = x)) %>%
ggplot(aes(x = value)) +
  geom_histogram(bins = 19, fill = "#03336B") +
  facet_wrap(~name, scales = "free", ncol = 3) +
  labs(y = element_blank()) +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

# Barplots
prep %>%
  select_at(x_discrete) %>%
  pivot_longer(everything()) %>%
  mutate(name = factor(name, levels = x_discrete),
         value = factor(value)) %>%
ggplot(aes(x = value)) +
  geom_bar(fill = "#03336B") +
  facet_wrap(~name, scales = "free", ncol = 1) +
  labs(y = "Count")

#====================================================================
# Bivariate description
#====================================================================

# Bivariate correlations
prep %>%
  select_at(c(y_var, x_continuous)) %>%
  cor() %>%
  round(2) %>%
ggcorrplot(
  hc.order = FALSE,
  type = "upper",
  outline.col = "white",
  ggtheme = ggplot2::theme_minimal(),
  colors = c("#6D9EC1", "white", "#E46726")
) + theme(plot.margin=grid::unit(c(0, 0, 0, 0), "mm"))

# log price ~ discrete
prep %>%
  select(all_of(x_discrete), y_var) %>%
  mutate(across(-y_var, as.factor)) %>%
  pivot_longer(cols = -y_var) %>%
ggplot(aes_string("value", y_var)) +
  geom_boxplot(varwidth = TRUE, color = "#03336B") +
  facet_wrap(~ name, scales = "free_x", ncol = 1) +
  scale_y_log10()

# log price ~ continuous
prep %>%
  select(all_of(x_continuous), y_var) %>%
  pivot_longer(cols = -y_var) %>%
ggplot(aes_string("value", y = y_var)) +
  geom_hex(bins = 32) +
  facet_wrap(~ name, scales = "free", ncol = 4) +
  scale_y_log10() +
  scale_fill_viridis_c(option = "magma", trans = "log10") +
  theme(legend.position = c(0.7, 0.1),
        legend.direction = "horizontal")

save(prep, x_continuous, x_coord, y_var, x_discrete, x_vars,
     file = "prep.RData")
