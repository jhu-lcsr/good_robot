// https://github.com/arunoda/meteor-smart-collections
//
// mrt add smart-collections
// now use Posts = new Meteor.SmartCollection('posts');
//
// authored by orefalo

/// <reference path="meteor.d.ts"/>

declare module Meteor {

	function SmartCollection<T>(name:string, options?: {
		connection?: Object;
		idGeneration?: Mongo.IdGenerationEnum;
		transform?: (document)=>any;
	}): void;

	interface SmartCollection<T> extends Mongo.Collection<T> {

		new(name:string, options?: {
			connection?: Object;
			idGeneration?: Mongo.IdGenerationEnum;
			transform?: (document)=>any;
		}):T;

		// keys can only be strings at this time - per author
		ObjectID(hexString?:string);

	}
}

declare module Mongo {
	enum IdGenerationEnum {
		STRING,
		MONGO
	}
}