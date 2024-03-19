module GalaxyPCA
end

using VirtualObservatory

gaia_sample_raw = execute(TAPService(:vizier), """select * from "I/355/gaiadr3" where RandomI < 10000"""; unitful=false)

using AstroQuery, FITSIO, WCS, HTTP, PyPlot, PyCall

# Catalogue name in VizieR
CATALOGUE = "I/239"

# Query the catalogue
catalogue_ivoid = "ivo://CDS.VizieR/$CATALOGUE"
resource = AstroQuery.vo.registry_search(ivoid=catalogue_ivoid)[1]

# Print metadata information about the catalogue
AstroQuery.describe(resource, verbose=true)

# Inspect in details the object and access the attributes
resource.creators[1]

# Get the tables available in the catalogue
tables = AstroQuery.get_tables(resource)
tables_names = keys(tables)

# Get the first table of the catalogue
first_table_name = first(tables_names)

# Execute a synchronous ADQL query
tap_service = AstroQuery.get_service(resource, "tap")
tap_records = AstroQuery.run_sync(tap_service, "select TOP 10 * from \"$first_table_name\"")
tap_records

# Execute a cone search query
conesearch_radius = 1 / 60.0  # in degrees
conesearch_center = (45.0439453125, 0.24246333228778008)
conesearch_records = AstroQuery.conesearch(resource, pos=conesearch_center, sr=conesearch_radius)
conesearch_records

# Get the catalogue coverage
moc_url = "https://cdsarc.cds.unistra.fr/viz-bin/moc/$CATALOGUE"
moc_fits = HTTP.get(moc_url).body
FITSIO.writeto("moc.fits", moc_fits)
catalogue_coverage = FITSIO.FITS("moc.fits")

# Plot the coverage with matplotlib
wcs = WCS.from_header(catalogue_coverage[1].header)
fig = PyPlot.figure(figsize=(5, 5))
ax = fig.add_subplot(projection=wcs)