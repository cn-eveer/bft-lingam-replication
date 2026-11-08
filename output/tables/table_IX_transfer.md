| Setting           | Task    | Model             | In-distribution macro-F1 | Transfer macro-F1 | Drop   | Transfer accuracy |
| ----------------- | ------- | ----------------- | ------------------------ | ----------------- | ------ | ----------------- |
| Core -> Product   | 4class  | RF                | 0.521                    | 0.503             | 0.019  | 0.512             |
| Core -> Product   | 4class  | LR                | 0.383                    | 0.419             | -0.036 | 0.449             |
| Core -> Product   | 4class  | BN                | 0.482                    | 0.44              | 0.042  | 0.468             |
| Core -> Product   | 4class  | LiNGAM-BN         | 0.338                    | 0.352             | -0.014 | 0.365             |
| Core -> Product   | 4class  | Refined LiNGAM-BN | 0.462                    | 0.459             | 0.003  | 0.479             |
| Core -> Product   | binary7 | RF                | 0.751                    | 0.73              | 0.022  | 0.73              |
| Core -> Product   | binary7 | LR                | 0.678                    | 0.683             | -0.004 | 0.691             |
| Core -> Product   | binary7 | BN                | 0.731                    | 0.719             | 0.012  | 0.725             |
| Core -> Product   | binary7 | LiNGAM-BN         | 0.615                    | 0.63              | -0.014 | 0.641             |
| Core -> Product   | binary7 | Refined LiNGAM-BN | 0.71                     | 0.71              | 0.001  | 0.718             |
| Product -> Core   | 4class  | RF                | 0.538                    | 0.483             | 0.055  | 0.485             |
| Product -> Core   | 4class  | LR                | 0.434                    | 0.353             | 0.08   | 0.391             |
| Product -> Core   | 4class  | BN                | 0.444                    | 0.427             | 0.017  | 0.444             |
| Product -> Core   | 4class  | LiNGAM-BN         | 0.348                    | 0.241             | 0.107  | 0.288             |
| Product -> Core   | 4class  | Refined LiNGAM-BN | 0.438                    | 0.43              | 0.007  | 0.438             |
| Product -> Core   | binary7 | RF                | 0.741                    | 0.725             | 0.017  | 0.727             |
| Product -> Core   | binary7 | LR                | 0.647                    | 0.621             | 0.026  | 0.641             |
| Product -> Core   | binary7 | BN                | 0.702                    | 0.702             | -0.0   | 0.709             |
| Product -> Core   | binary7 | LiNGAM-BN         | 0.603                    | 0.525             | 0.078  | 0.564             |
| Product -> Core   | binary7 | Refined LiNGAM-BN | 0.666                    | 0.681             | -0.015 | 0.686             |
| Core, temporal    | 4class  | RF                | 0.538                    | 0.538             | -0.0   | 0.529             |
| Core, temporal    | 4class  | LR                | 0.434                    | 0.313             | 0.121  | 0.38              |
| Core, temporal    | 4class  | BN                | 0.444                    | 0.502             | -0.058 | 0.493             |
| Core, temporal    | 4class  | LiNGAM-BN         | 0.348                    | 0.352             | -0.004 | 0.359             |
| Core, temporal    | 4class  | Refined LiNGAM-BN | 0.438                    | 0.502             | -0.064 | 0.493             |
| Core, temporal    | binary7 | RF                | 0.741                    | 0.723             | 0.018  | 0.737             |
| Core, temporal    | binary7 | LR                | 0.647                    | 0.543             | 0.104  | 0.622             |
| Core, temporal    | binary7 | BN                | 0.702                    | 0.743             | -0.041 | 0.752             |
| Core, temporal    | binary7 | LiNGAM-BN         | 0.603                    | 0.62              | -0.018 | 0.639             |
| Core, temporal    | binary7 | Refined LiNGAM-BN | 0.666                    | 0.724             | -0.058 | 0.734             |
| Product, temporal | 4class  | RF                | 0.521                    | 0.545             | -0.024 | 0.54              |
| Product, temporal | 4class  | LR                | 0.383                    | 0.357             | 0.026  | 0.418             |
| Product, temporal | 4class  | BN                | 0.482                    | 0.507             | -0.024 | 0.521             |
| Product, temporal | 4class  | LiNGAM-BN         | 0.338                    | 0.288             | 0.051  | 0.344             |
| Product, temporal | 4class  | Refined LiNGAM-BN | 0.462                    | 0.539             | -0.077 | 0.541             |
| Product, temporal | binary7 | RF                | 0.751                    | 0.769             | -0.018 | 0.774             |
| Product, temporal | binary7 | LR                | 0.678                    | 0.641             | 0.037  | 0.68              |
| Product, temporal | binary7 | BN                | 0.731                    | 0.697             | 0.034  | 0.7               |
| Product, temporal | binary7 | LiNGAM-BN         | 0.615                    | 0.558             | 0.057  | 0.625             |
| Product, temporal | binary7 | Refined LiNGAM-BN | 0.71                     | 0.744             | -0.034 | 0.758             |
