default: edit_on_local_serve

edit_on_local_serve: clean
	docker-compose up

# This doesn't work right now
server_html:
	docker-compose run notebookserver ipython nbconvert --profile techfest techfest_tutorial_intro.ipynb --to slides --post serve

dump_presentation_html: clean
	docker-compose run notebookserver ipython nbconvert techfest_tutorial_intro.ipynb --to slides

clean:
	find . -type f -name "*.pyc" -delete

copy_presentation_html: dump_presentation_html
	cp notebooks/techfest_tutorial_intro.slides.html presentation/intro_presentation.html

dev_presentation: copy_presentation_html
	cd presentation && rm -f media custom.css index.html
	cd presentation && ln -s ../intro/media media 
	cd presentation && ln -s ../intro/static/foundation-5.5.1.css custom.css
	cd presentation && ln -s ../notebooks/techfest_tutorial_intro.slides.html index.html
	cd presentation && python -m SimpleHTTPServer 8000

presentation_80: copy_presentation_html
	cd presentation
	sudo python -m SimpleHTTPServer 80

multiserver: buildmultiserver runmultiserver

buildmultiserver:
	docker-compose -f multy_compose.yml build

runmultiserver:
	docker-compose -f multy_compose.yml up -d

stopmultiserver:
	docker-compose -f multy_compose.yml stop
