CMAP: headers
	g++ -g -std=c++11 CMAP.cpp -o CMAP -pthread

headers:
	g++ -std=c++11 CMAP.h
	g++ -std=c++11 utils.h
	g++ -std=c++11 CMAP_UNIFIED.h
	g++ -std=c++11 CMAP_UNIFIED_HR.h
	g++ -std=c++11 CMAP_FACTORED.h
	g++ -std=c++11 CMAP_FACTORED_HR.h
	g++ -std=c++11 CMAP_FACTORED_TH.h
	g++ -std=c++11 CMAP_FACTORED_HR_TH.h

clean:
	rm *.gch CMAP