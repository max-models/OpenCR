// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/OpenCR/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'OpenCR',
	title: 'OpenCR',
	description: 'Collisional-radiative models of low-temperature plasmas in Python.',
	pythonPackage: 'opencr',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Guides', link: '/guides/model/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Full examples', link: '/full-examples/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/opencr/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Guides',
			items: [
				{ label: 'The CR model', slug: 'guides/model' },
				{ label: 'Atomic hydrogen', slug: 'guides/atomic-hydrogen' },
				{ label: 'Molecular hydrogen', slug: 'guides/molecular-hydrogen' },
				{ label: 'Argon and metals', slug: 'guides/argon-and-metals' },
				{ label: 'Sensitivities and inference', slug: 'guides/inference' },
				{ label: 'Data and provenance', slug: 'guides/data' },
				{ label: 'Solvers and troubleshooting', slug: 'guides/solvers' },
				{ label: 'Verification', slug: 'guides/verification' },
				{ label: 'Backends', slug: 'guides/backends' },
				{ label: 'Benchmarks', slug: 'guides/benchmarks' },
			],
		},
		{
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Full examples',
			items: [{ autogenerate: { directory: 'full-examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
