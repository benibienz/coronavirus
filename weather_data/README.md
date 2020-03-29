<b>Values of -1000 are dummy values for when there was some sort of error in the JSON response.<br>
This will eventually be investigated and corrected. The -1000 values are not valid and should be ignored for now.</b><br><br>
Please see https://darksky.net/dev/docs for description of the various fields that are captured.<br>
All of the following details pertain to data from the 'daily' element in the JSON file that is returned.<br>
Where applicable, the units that are returned are in SI units<br><br>
The data is all very generously Powered by Dark Sky: https://darksky.net/poweredby/<br><br>

The following csv files are included:<br><br>
<b>tMax.csv<br></b>
'temperatureHigh' field, provided as degrees Celsius<br><br>

<b>tMin.csv<br></b>
'temperatureLow' field, provided as degrees Celsius<br><br>

<b>humidity.csv<br></b>
'humidity' field, provided as relative humidity (percent out of 100)<br><br>

<b>uv.csv<br></b>
'uvIndex' field, provided as UV index. Honestly, I'm not sure what the units here are.<br><br>

<b>cloud.csv<br></b>
'cloudCover' field, provided as The percentage of sky occluded by clouds (out of 100)<br><br>

<b>precip.csv<br></b>
'precipProbability' field, provided as the probability of precipitation occurring (out of 100)<br><br>

<b>dew.csv<br></b>
'dewPoint' field, provided as degrees Fahrenheit (according to the documentation) I made the request for SI units<br>
Frankly, I don't know enough about this to know if the numbers look like Fahrenheit or Celsius<br><br>

<b>pressure.csv<br></b>
'pressure" field, provided as sea-level air pressure in millibars<br><br>

<b>wind.csv</b><br>
'windSpeed' field, provided as wind speed in miles per hour (according to the documentation). Again as above,<br>
I made the request for SI units. I don't know enough to know which one is more likely.<br><br>

<b>ozone.csv<br></b>
'ozone' field, provided as columnar density of total atmospheric ozone at the given time in Dobson units<br><br>

<b>sunrise.csv</b><br>
'sunriseTime' field, provided as Unix time (https://en.wikipedia.org/wiki/Unix_time)<br><br>

<b>sunset.csv</b><br>
'sunsetTime' field, provided as Unix time<br><br>
