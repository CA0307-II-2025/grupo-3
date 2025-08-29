library(readr)
library(tidyverse)

datos <- read_csv("../data/clean/dashboard_2010_clean.csv")

datos

colSums(is.na(datos))


library(dplyr)
library(tidyr)
library(knitr)
library(kableExtra)

datos %>%
  select(-"OPE ID") %>%
  summarise(across(where(is.numeric),
                   list(
                     minimo = ~min(.x, na.rm = TRUE),
                     Q1 = ~quantile(.x, 0.25, na.rm = TRUE),
                     Q2 = ~quantile(.x, 0.50, na.rm = TRUE),
                     Q3 = ~quantile(.x, 0.75, na.rm = TRUE),
                     maximo = ~max(.x, na.rm = TRUE),
                     promedio = ~round(mean(.x, na.rm = TRUE), 2),
                     na = ~sum(is.na(.x))
                   ))) %>%
  pivot_longer(everything(),
               names_to = c("Variable", ".value"),
               names_sep = "_") %>%
  kable(caption = "Resumen estadístico de variables numéricas") %>%
  kable_styling(full_width = FALSE,
                bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center")
