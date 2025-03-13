def data_validation(returns):
    """
    Verifica la consistencia de los datos antes de la optimización.
    """

    print("Resumen estadístico de los retornos:")
    print(returns.describe())
    print("\nCorrelación entre activos:")
    print(returns.corr()) # type: ignore #

