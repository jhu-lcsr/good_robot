/**========================================================
 * Module: clearicon.js
 * Created by wjwong on 9/8/15.
 =========================================================*/

angular.module('angle').directive('clearIcon', ['$compile',
 function($compile) {
  return {
   restrict : 'A',
   link : function(scope, elem, attrs) {
    var model = attrs.ngModel;
    //form-control-feedback
    var template = '<span ng-show="'+model+'" ng-click=\"' + model + '=\'\'\" class="form-control-feedback fa fa-times-circle" style="pointer-events: auto; top: 5px; left: -4px;"></span>';
    elem.parent().append($compile(template)(scope));
    var clearIconToggle = function(toggleParam) {
     if(elem.val().trim().length)
      elem.next().css("display", "inline");
     else {
      if(elem.next().css("display") == "inline")
       elem.next().css("display", "none");
     }
    };
    /*var clearText = function(clearParam) {
     elem.val('');
     clearIconToggle(clearParam);
     };*/
    elem.on("focus", function(event) {
     clearIconToggle(model);
    });
    elem.on("keyup", function(event) {
     clearIconToggle(model);
    });
    elem.next().on("click", function(event) {
     console.warn('click', event);
     elem.val('');
     elem.next().css("display", "none");
    });
   }
  }; }]);