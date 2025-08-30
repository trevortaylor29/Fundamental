# fundintel/universe.py
DEFAULT_UNIVERSE = [
    # ---- Core US equity (large/mid/small; multi-issuer)
    "SPY","IVV","VOO","SPLG","VTI","SCHB","ITOT","IWB","IWM","IJH","IJR","VO","VB",
    # ---- Equal-weight & alt core
    "RSP","QQQ","QQQM","DIA","VUG","VTV","IWF","IWD","MGK","VBR","VOT","AVUS",
    # ---- Sectors (SPDR + alternates)
    "XLK","XLF","XLE","XLY","XLP","XLV","XLI","XLU","XLRE","XLC","XLB",
    "VDE","VHT","VCR","VDC","VIS","VAW","VPU","VNQ","VOX",
    # ---- Factor / dividend / low vol
    "SCHD","VYM","DVY","HDV","NOBL","QUAL","MTUM","USMV","SPLV","SPHB","VLUE",
    # ---- International developed & EM (broad + region + country)
    "VEA","VXUS","IXUS","EFA","IEFA","EEM","IEMG","VWO","SCZ","EWC","EWA","EWJ",
    "EWG","EWU","EWL","EWQ","EWP","EWT","EWY","EWH","INDA","EWZ","ARGT","EZA",
    # ---- Real estate
    "VNQ","IYR","SCHH","XLRE",
    # ---- Commodities / gold & silver
    "GLD","IAU","SGOL","SLV","DBC","PDBC","GSG",
    # ---- Bonds / rates / credit / cash-like
    "AGG","BND","SCHZ","IUSB","LQD","VCIT","HYG","JNK",
    "SHY","IEF","IEI","TLT","ZROZ","VTIP","TIP","STIP",
    "SGOV","BIL","SHV",
    # ---- Thematics & semis/biotech/clean energy/uranium
    "ARKK","ARKW","SOXX","SMH","XBI","IBB","TAN","ICLN","URA","CIBR","BOTZ",
    # ---- Leveraged & inverse (penalized by Structure)
    "TQQQ","SQQQ","UPRO","SPXU","SOXL","SOXS","FNGU","FNGD",
    # ---- Popular mutual funds (Vanguard/Fidelity/Schwab)
    "VTSAX","VFIAX","VBTLX","VWNAX","VEMAX","FXAIX","FSKAX","SWPPX","SWISX","SWTSX",
    # ---- Closed-end funds (sample)
    "PDI","PDO","UTF","UTG","BST",
    # ---- Others
    "^GSPC", "USD"
]

