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
	find presentation.html

presentation: dump_presentation_html 
	cp notebooks/techfest_tutorial_intro.slides.html presentation.html
	python -m SimpleHTTPServer 8000
