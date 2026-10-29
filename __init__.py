"""SAMStars Processing tools for QGIS."""


def classFactory(iface):
    from .samstars_plugin import SamStarsPlugin

    return SamStarsPlugin(iface)
