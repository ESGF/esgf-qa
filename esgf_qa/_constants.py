from datetime import timedelta

# Mapping of checker names to project names for better readability
checker_dict = {
    "cc6": "CORDEX-CMIP6",
    "cf": "CF-Conventions",
    "mip": "MIP",
    # "wcrp-cmip5": "CMIP5",
    "wcrp_cmip6": "CMIP6",
    "wcrp_cmip6plus": "CMIP6Plus",
    # "wcrp_cmip7aft: "CMIP7-AFT",
    "wcrp_cmip7": "CMIP7",
    # "wcrp_cordex": "CORDEX",
    "wcrp_cordex_cmip6": "CORDEX-CMIP6",
    # "obs4mips": "Obs4MIPs",
    # "input4mips": "Input4MIPs",
}
checker_dict_ext = {
    # "pcons": "ParentConsistency"
    "cons": "Consistency",
    "cont": "Continuity",
    "comp": "Compatibility",
    **checker_dict,
}
# Checkers for which consistency checks should be run
checker_supporting_consistency_checks = [
    "wcrp_cmip7",
    "wcrp_cmip6",
    "wcrp_cmip6plus",
    "wcrp_cordex_cmip6",
    "cc6",
    "mip",
]

# DRS parent directory names (for identifying project root and building dataset id)
supported_project_ids = [
    "cmip7",
    "cmip6plus",
    "cmip6",
    "cmip5",
    "cordex",
    "cordex-cmip6",
    "cordex-fpsconv",
    "obs4mips",
    "input4mips",
    "c3scordex",
    "c3scmip5",
    "c3scmip6",
    "c3s-ipcc-ar6-atlas",
    "c3satlas",
    "c3s-cica-atlas",
    "c3satlas_v1",
    "c3s-atlas-dataset",
    "c3satlas_v2",
    "eerie",
    "happi",
    "cosmo-rea",
]

# Definition of maximum permitted deviations from the given frequency
deltdic = {}
deltdic["monmax"] = timedelta(days=31.01).total_seconds()
deltdic["monmin"] = timedelta(days=27.99).total_seconds()
deltdic["mon"] = timedelta(days=31).total_seconds()
deltdic["daymax"] = timedelta(days=1.01).total_seconds()
deltdic["daymin"] = timedelta(days=0.99).total_seconds()
deltdic["day"] = timedelta(days=1).total_seconds()
deltdic["1hrmin"] = timedelta(hours=0.99).total_seconds()
deltdic["1hrmax"] = timedelta(hours=1.01).total_seconds()
deltdic["1hr"] = timedelta(hours=1).total_seconds()
deltdic["3hrmin"] = timedelta(hours=2.99).total_seconds()
deltdic["3hrmax"] = timedelta(hours=3.01).total_seconds()
deltdic["3hr"] = timedelta(hours=3).total_seconds()
deltdic["6hrmin"] = timedelta(hours=5.99).total_seconds()
deltdic["6hrmax"] = timedelta(hours=6.01).total_seconds()
deltdic["6hr"] = timedelta(hours=6).total_seconds()
deltdic["yrmax"] = timedelta(days=366.01).total_seconds()
deltdic["yrmin"] = timedelta(days=359.99).total_seconds()
deltdic["yr"] = timedelta(days=360).total_seconds()
deltdic["subhr"] = timedelta(seconds=600).total_seconds()
deltdic["subhrmax"] = timedelta(seconds=601).total_seconds()
deltdic["subhrmin"] = timedelta(seconds=599).total_seconds()
deltdic["dec"] = timedelta(days=3600).total_seconds()
deltdic["decmax"] = timedelta(days=3660.01).total_seconds()
deltdic["decmin"] = timedelta(days=3599.99).total_seconds()
deltdic["cen"] = timedelta(days=36000).total_seconds()
deltdic["cenmax"] = timedelta(days=36600.01).total_seconds()
deltdic["cenmin"] = timedelta(days=35999.99).total_seconds()
# CMIP-style frequencies for "time: point":
for l_freq in ["subhr", "1hr", "3hr", "6hr", "day", "mon", "yr"]:
    deltdic[l_freq + "Pt"] = deltdic[l_freq]
    deltdic[l_freq + "Ptmax"] = deltdic[l_freq + "max"]
    deltdic[l_freq + "Ptmin"] = deltdic[l_freq + "min"]
