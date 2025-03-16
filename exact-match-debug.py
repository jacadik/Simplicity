# Add this debug code to the app.py file
# Insert this right after the get_similar_paragraphs call in the view_similarity route

@app.route('/similarity')
def view_similarity():
    """View similarity analysis with threshold adjustment."""
    threshold = request.args.get('threshold', type=float, default=0.8)
    
    # Get similarity results
    similarities = db_manager.get_similar_paragraphs(threshold)
    
    # DEBUG: Count exact matches
    exact_count = sum(1 for sim in similarities if sim['similarity_type'] == 'exact')
    similar_count = sum(1 for sim in similarities if sim['similarity_type'] == 'similar')
    app.logger.info(f"DEBUG: Displaying {exact_count} exact matches and {similar_count} similar paragraphs")
    
    # DEBUG: Check database directly for exact matches
    session = db_manager.Session()
    try:
        # Direct query to count exact matches
        exact_db_count = session.query(SimilarityResult).filter(
            SimilarityResult.similarity_type == 'exact'
        ).count()
        app.logger.info(f"DEBUG: Database contains {exact_db_count} exact match records")
    except Exception as e:
        app.logger.error(f"Error in debug query: {str(e)}")
    finally:
        session.close()
    
    return render_template('similarity.html', similarities=similarities, threshold=threshold)


# Add this to the analyze_similarity route to log what's being added
@app.route('/analyze-similarity', methods=['POST'])
def analyze_similarity():
    """Run similarity analysis on paragraphs."""
    threshold = float(request.form.get('threshold', 0.8))
    
    # Get all paragraphs
    paragraphs = db_manager.get_paragraphs()
    
    if not paragraphs:
        flash('No paragraphs available for analysis', 'warning')
        return redirect(url_for('view_similarity'))
    
    # Prepare data for similarity analysis
    para_data = []
    for para in paragraphs:
        para_data.append({
            'id': para['id'],
            'content': para['content'],
            'doc_id': para['document_id']
        })
    
    # Find exact matches first
    exact_matches = similarity_analyzer.find_exact_matches(para_data)
    app.logger.info(f"DEBUG: Found {len(exact_matches)} exact matches during analysis")
    
    # Find similar paragraphs
    similar_paragraphs = similarity_analyzer.find_similar_paragraphs(para_data, threshold)
    app.logger.info(f"DEBUG: Found {len(similar_paragraphs)} similar paragraphs during analysis")
    
    # Combine results
    all_results = exact_matches + similar_paragraphs
    
    if all_results:
        # Save results to database
        db_manager.add_similarity_results(all_results)
        flash(f'Found {len(exact_matches)} exact matches and {len(similar_paragraphs)} similar paragraphs', 'success')
    else:
        flash('No similarities found', 'info')
    
    return redirect(url_for('view_similarity', threshold=threshold))
