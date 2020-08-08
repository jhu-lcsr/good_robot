
HTTP.methods({

  '/translate/:lang': function() {
    var tr = Assets.getText('i18n/'+ this.params.lang);
    return tr;
  }

});

